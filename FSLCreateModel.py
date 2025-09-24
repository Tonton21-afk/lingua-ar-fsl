import os, cv2, time, math, random
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
import re
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def natural_key(s: str):
    # Example: "10" -> [10], "A" -> ["a"], "2B" -> [2, "b"]
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    pose = (np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],
                     dtype=np.float32).flatten() if results.pose_landmarks else np.zeros(33*4, dtype=np.float32))
    face = (np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark],
                     dtype=np.float32).flatten() if results.face_landmarks else np.zeros(468*3, dtype=np.float32))
    lh = (np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark],
                   dtype=np.float32).flatten() if results.left_hand_landmarks else np.zeros(21*3, dtype=np.float32))
    rh = (np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark],
                   dtype=np.float32).flatten() if results.right_hand_landmarks else np.zeros(21*3, dtype=np.float32))
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)

class SignLanguageProcessor:
    def __init__(self, data_path, mp_data_path):
        self.data_path = data_path
        self.mp_data_path = mp_data_path
        self.actions = []
        self.model = None
        self.sequence_length = 30

    def detect_actions_from_folders(self):
        self.actions = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        log(f"Detected actions: {self.actions}")
        return np.array(self.actions)

    def setup_data_folders(self):
        os.makedirs(self.mp_data_path, exist_ok=True)
        self.actions = self.detect_actions_from_folders()
        for action in self.actions:
            for pos in range(self.sequence_length):
                os.makedirs(os.path.join(self.mp_data_path, action, str(pos)), exist_ok=True)
        log(f"Setup completed for {len(self.actions)} actions")
        return True


    def process_videos(self):
        if not self.actions:
            self.detect_actions_from_folders()
        self.actions = sorted(self.actions, key=natural_key)

        for action in self.actions:
            video_dir = os.path.join(self.data_path, action)
            data_dir  = os.path.join(self.mp_data_path, action)
            if not os.path.isdir(video_dir):
                log(f"Warning: Missing video directory for action '{action}', skipping.")
                continue

            video_files = sorted(
                [f for f in os.listdir(video_dir)
                if f.lower().endswith(('.mp4','.avi','.mov','.mkv','.webm'))],
                key=natural_key
            )
            log(f"Processing {len(video_files)} videos for {action}")

            # one Holistic per action (faster)
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as holistic:

                for idx, vf in enumerate(video_files[:1000]):
                    cap_path = os.path.join(video_dir, vf)
                    cap = cv2.VideoCapture(cap_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames == 0:
                        cap.release()
                        continue

                    frame_idx = np.linspace(0, total_frames-1, self.sequence_length, dtype=int)
                    for pos, fi in enumerate(frame_idx):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        _, results = mediapipe_detection(frame, holistic)
                        keypoints = extract_keypoints(results)
                        np.save(os.path.join(data_dir, str(pos), f"{idx}.npy"), keypoints)

                    cap.release()

                    if (idx + 1) % 100 == 0:
                        log(f"{action}: processed {idx+1}/{min(len(video_files),1000)} videos")

        log("Finished processing all videos!")
        return True

    def prepare_training_data(self):
        if len(self.actions) == 0:
            self.detect_actions_from_folders()
            if len(self.actions) == 0:
                log("No actions found for training!")
                return None, None, None, None

        # 🔥 ensure deterministic, numeric-aware order
        self.actions = sorted(self.actions, key=natural_key)

        # 🔥 persist actions order for inference/reload
        order_path = os.path.join(self.mp_data_path, "actions_order.txt")
        with open(order_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.actions))

        label_map = {label: i for i, label in enumerate(self.actions)}

        action_to_ids, total_expected = {}, 0
        for action in self.actions:
            ids = self._discover_ids_for_action(action)
            action_to_ids[action] = ids
            total_expected += len(ids)

        if total_expected == 0:
            log("No training data found! Run process_videos first.")
            return None, None, None, None

        X_path = os.path.join(self.mp_data_path, "X_memmap.npy")
        y_path = os.path.join(self.mp_data_path, "y_memmap.npy")

        def _allocate_new_memmaps():
            X_mm = np.lib.format.open_memmap(
                X_path, dtype=np.float32, mode='w+',
                shape=(total_expected, self.sequence_length, 1662)
            )
            y_mm = np.lib.format.open_memmap(
                y_path, dtype=np.int32, mode='w+',
                shape=(total_expected,)
            )
            y_mm[:] = -1
            return X_mm, y_mm

        rebuild = False
        if os.path.exists(X_path) and os.path.exists(y_path):
            try:
                X = np.load(X_path, mmap_mode="r+")
                y = np.load(y_path, mmap_mode="r+")
                if X.shape != (total_expected, self.sequence_length, 1662) or y.shape != (total_expected,):
                    log("Memmap shapes mismatch current dataset — rebuilding memmaps...")
                    rebuild = True
                else:
                    if np.all(y >= 0):
                        log("Dataset already prepared — skipping Step 3 (using existing memmaps).")
                        y_cat = to_categorical(y, num_classes=len(self.actions))
                        idxs = np.arange(total_expected)
                        train_idx, test_idx = train_test_split(idxs, test_size=0.05, random_state=42)
                        return X_path, y_cat, train_idx, test_idx
                    else:
                        log("Resuming from existing (partially filled) memmaps...")
            except Exception as e:
                log(f"Could not open existing memmaps ({e}) — rebuilding...")
                rebuild = True

            if rebuild:
                try: os.remove(X_path)
                except Exception: pass
                try: os.remove(y_path)
                except Exception: pass
                X, y = _allocate_new_memmaps()
        else:
            X, y = _allocate_new_memmaps()

        try:
            start_idx = int(np.argmax(y == -1)) if np.any(y == -1) else y.shape[0]
        except Exception:
            start_idx = int(np.count_nonzero(y >= 0))

        idx = start_idx
        for action, ids in action_to_ids.items():
            n = len(ids)
            log(f"Preparing data for {n} videos of action: {action}")
            for i, vid in enumerate(ids):
                if idx < y.shape[0] and y[idx] != -1:
                    idx += 1
                    continue

                window = []
                for pos in range(self.sequence_length):
                    npy_path = os.path.join(self.mp_data_path, action, str(pos), f"{vid}.npy")
                    try:
                        res = np.load(npy_path, allow_pickle=False).astype(np.float32, copy=False)
                        if res.size != 1662:
                            fixed = np.zeros(1662, dtype=np.float32)
                            fixed[:min(1662, res.size)] = res[:min(1662, res.size)]
                            res = fixed
                    except Exception:
                        res = np.zeros(1662, dtype=np.float32)
                    window.append(res)

                X[idx] = np.stack(window, axis=0)
                y[idx] = label_map[action]
                idx += 1

                if ((i + 1) % 100 == 0) or ((i + 1) == n):
                    log(f"  Processed {i+1}/{n} videos")

        y_cat = to_categorical(y, num_classes=len(self.actions))
        idxs = np.arange(total_expected)
        train_idx, test_idx = train_test_split(idxs, test_size=0.05, random_state=42)

        log(f"Prepared {total_expected} sequences across {len(self.actions)} actions")
        return X_path, y_cat, train_idx, test_idx



    def build_model(self):
        if len(self.actions) == 0:
            self.detect_actions_from_folders()
            if len(self.actions) == 0:
                log("No actions found for model building!")
                return None
        
        log(f"Building model for {len(self.actions)} actions")
        
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length, 1662)),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(self.actions), activation='softmax')
        ])
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model = model
        log("Model built successfully")
        return model

    def train_model(self, X_path, y_cat, train_idx, test_idx, epochs=200, batch_size=32):
        X_mem = np.load(X_path, mmap_mode='r')
        y_cat = np.asarray(y_cat)
        seq_len  = X_mem.shape[1]
        feat_dim = X_mem.shape[2]

        class DiskSequence(Sequence):
            def __init__(self, indices, batch_size, seq_len, feat_dim):
                self.indices = np.asarray(indices, dtype=np.int64)
                self.batch_size = int(batch_size)
                self.seq_len = int(seq_len)
                self.feat_dim = int(feat_dim)
                self.n = self.indices.shape[0]
                self.on_epoch_end()

            def __len__(self):
                return math.ceil(self.n / self.batch_size)

            def __getitem__(self, idx):
                start = idx * self.batch_size
                end = min(start + self.batch_size, self.n)
                batch_inds = self.indices[start:end]

                bx = np.empty((len(batch_inds), self.seq_len, self.feat_dim), dtype=np.float32)
                for j, ix in enumerate(batch_inds):
                    bx[j] = X_mem[ix]

                by = y_cat[batch_inds]
                return bx, by

            def on_epoch_end(self):
                np.random.shuffle(self.indices)

        train_seq = DiskSequence(train_idx, batch_size, seq_len, feat_dim)
        val_seq   = DiskSequence(test_idx,  batch_size, seq_len, feat_dim)

        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.mp_data_path, 'logs', run_stamp)
        os.makedirs(log_dir, exist_ok=True)

        tb = TensorBoard(log_dir=log_dir)
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            os.path.join(self.mp_data_path, 'best_fsl_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

        log(f"Training started (epochs={epochs}, batch_size={batch_size}) — logs at {log_dir}")
        hist = self.model.fit(
            train_seq,
            epochs=epochs,
            validation_data=val_seq,
            callbacks=[tb, checkpoint],
            verbose=1,
            workers=0,
            use_multiprocessing=False,
            max_queue_size=10
        )
        log("Training complete")
        return hist



    def evaluate_model(self, X_path, y_cat, test_idx, batch_size=32):
        X = np.load(X_path, mmap_mode='r')
        y_true = []
        y_pred = []

        for start in range(0, len(test_idx), batch_size):
            end = start + batch_size
            batch_idx = test_idx[start:end]

            X_batch = X[batch_idx]
            y_batch = y_cat[batch_idx]

            preds = self.model.predict(X_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(y_batch, axis=1))

        acc = accuracy_score(y_true, y_pred)
        log(f"Accuracy: {acc*100:.2f}%")
        return acc


    def save_model(self, model_name='sign_language_model.h5'):
        path = os.path.join(self.mp_data_path, model_name)
        self.model.save(path)
        log(f"Model saved at {path}")

    def load_model(self, model_name='sign_language_model.h5'):
        path = os.path.join(self.mp_data_path, model_name)
        self.model = load_model(path)
        order_path = os.path.join(self.mp_data_path, "actions_order.txt")
        if os.path.exists(order_path):
            with open(order_path, "r", encoding="utf-8") as f:
                self.actions = [ln.strip() for ln in f if ln.strip()]
            log(f"Loaded actions order ({len(self.actions)} classes).")
        else:
            self.detect_actions_from_folders()
            log("WARNING: actions_order.txt missing; using folder order (may mismatch).")
        # sanity: model outputs must match class count
        out_units = int(self.model.output_shape[-1])
        if out_units != len(self.actions):
            log(f"ERROR: model outputs {out_units} classes but actions has {len(self.actions)}. Rebuild with consistent order.")
        log(f"Model loaded from {path}")
        return self.model
    
    def dbg_check_model_vs_actions(self):
        if self.model is None:
            log("No model loaded."); return
        out_units = int(self.model.output_shape[-1])
        log(f"Model output units: {out_units}, actions: {len(self.actions)}")
        if out_units != len(self.actions):
            log("❌ Mismatch — labels/order problem.")

    def dbg_label_distribution(self):
        y_path = os.path.join(self.mp_data_path, "y_memmap.npy")
        if not os.path.exists(y_path):
            log("y_memmap.npy not found. Run 3.Prepare first."); return
        y = np.load(y_path, mmap_mode="r")
        vals, cnts = np.unique(y, return_counts=True)
        log(f"Labels present: {vals.tolist()}")
        log(f"Counts per label: {cnts.tolist()}")

    def dbg_data_health(self):
        X_path = os.path.join(self.mp_data_path, "X_memmap.npy")
        if not os.path.exists(X_path):
            log("X_memmap.npy not found. Run 3.Prepare first."); return
        X = np.load(X_path, mmap_mode="r")
        nz = np.count_nonzero(X)
        frac = nz / X.size
        log(f"X shape: {X.shape}, mean={float(X.mean()):.4f}, std={float(X.std()):.4f}, nonzero_frac={frac:.6f}")


    
    def run_realtime(self, model_path='None', prob_threshold=0.60, show_fps=True):
        if self.model is None:
            if model_path:
                try:
                    self.model = load_model(model_path)
                    log(f"Loaded model for realtime from: {model_path}")
                except Exception as e:
                    log(f"Failed to load model at {model_path}: {e}")
                    return
            else:
                log("No model in memory. Load or train a model first.")
                return

        if not self.actions:
            self.detect_actions_from_folders()
            if not self.actions:
                log("No actions found. Cannot label predictions.")
                return

        from collections import deque
        window = deque(maxlen=self.sequence_length)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log("Cannot open webcam.")
            return

        prev_time = time.time()
        fps = 0.0

        # smoothing + logging state
        pred_hist = deque(maxlen=5)
        last_log_t = 0.0

        # allow live threshold tuning
        thr = float(prob_threshold)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            log("Realtime recognition started. Press 'q' to quit. ('t' - thr -0.05, 'y' +0.05)")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                window.append(keypoints)

                # draw landmarks (unchanged functionality)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # FPS (unchanged)
                if show_fps:
                    now = time.time()
                    dt = now - prev_time
                    prev_time = now
                    fps = 1.0 / dt if dt > 0 else fps
                    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # warm-up: wait until we have a full window
                if len(window) < self.sequence_length:
                    cv2.rectangle(image, (10, 50), (700, 95), (0, 0, 0), -1)
                    cv2.putText(image, f"Collecting frames... {len(window)}/{self.sequence_length}",
                                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.imshow("FSL Realtime Recognition", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue

                # ---- PREDICT ----
                seq = np.expand_dims(np.array(window, dtype=np.float32), axis=0)
                probs = self.model.predict(seq, verbose=0)[0]

                # sanitize probabilities
                if not np.isfinite(probs).all():
                    log("WARN: non-finite probabilities detected; sanitizing.")
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

                # top-k (3 on overlay, 5 in logs)
                order = np.argsort(probs)[::-1]
                topk = order[:3]
                topk_probs = probs[topk]

                # majority-vote smoothing for final label
                pred_hist.append(int(topk[0]))
                stable = max(set(pred_hist), key=pred_hist.count)
                stable_votes = pred_hist.count(stable)

                # final decision: need 3/5 votes AND pass threshold, else fallback to top-1 this frame
                final_idx = int(stable) if (stable_votes >= 3 and probs[stable] >= thr) else int(topk[0])
                final_conf = float(probs[final_idx])
                final_label = self.actions[final_idx] if final_conf >= thr else "Unsure"

                # ---- draw overlay (Top-3 + threshold indicator) ----
                cv2.rectangle(image, (10, 50), (1000, 170), (0, 0, 0), -1)
                cv2.putText(image, f"{final_label}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                for i, (cls_idx, p) in enumerate(zip(topk, topk_probs), start=1):
                    txt = f"{i}. {self.actions[int(cls_idx)]}: {p*100:.1f}%"
                    cv2.putText(image, txt, (20, 110 + (i-1)*22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.putText(image, f"thr={thr:.2f}", (920, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # ---- terminal logs every ~0.5s ----
                now_t = time.time()
                if now_t - last_log_t >= 0.5:
                    last_log_t = now_t
                    top5 = order[:5]
                    log(f"TOPK: {[(self.actions[int(i)], round(float(probs[i]),4)) for i in top5]} "
                        f"| final=({final_label}, {final_conf:.3f}) thr={thr:.2f}")

                # show frame
                cv2.imshow("FSL Realtime Recognition", image)

                # keys: quit / threshold up/down
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    thr = max(0.05, round(thr - 0.05, 2))
                elif key == ord('y'):
                    thr = min(0.95, round(thr + 0.05, 2))

        cap.release()
        cv2.destroyAllWindows()
        log("Realtime recognition ended.")


    
    def predict_one(self, action: str, vid_id: int):
        """
        Run inference on a single preprocessed sequence (no webcam).
        `action` is the folder name, `vid_id` is the numeric id (e.g., 0, 1, 2...) of that video.
        Prints Top-5 predictions with probabilities.
        """
        if self.model is None:
            log("Load/train a model first (option 8 or 5)."); return
        if not self.actions:
            self.detect_actions_from_folders()
        # sanity: output units must match class count
        out_units = int(self.model.output_shape[-1])
        if out_units != len(self.actions):
            log(f"ERROR: model outputs {out_units} classes but actions has {len(self.actions)}. "
                f"Check actions_order.txt alignment."); return
        
        # build the (1, T, F) sequence from saved npy frames
        seq = []
        for pos in range(self.sequence_length):
            path = os.path.join(self.mp_data_path, action, str(pos), f"{vid_id}.npy")
            if not os.path.exists(path):
                log(f"Missing frame file: {path}"); return
            arr = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            # pad/truncate safety
            feat_dim = int(self.model.input_shape[-1])
            if arr.size != feat_dim:
                fixed = np.zeros(feat_dim, np.float32)
                fixed[:min(feat_dim, arr.size)] = arr[:min(feat_dim, arr.size)]
                arr = fixed
            seq.append(arr)
        seq = np.expand_dims(np.stack(seq, axis=0), axis=0)  # (1, T, F)

        probs = self.model.predict(seq, verbose=0)[0]
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(probs)[::-1]
        top5 = order[:5]
        log(f"ONE SAMPLE → TOP5: {[(self.actions[int(i)], float(probs[i])) for i in top5]}")
        top1 = int(top5[0])
        log(f"TOP1: {self.actions[top1]} ({float(probs[top1])*100:.2f}%)")

    def rebuild_labels_only(self):
        """
        Rebuild y_memmap.npy to realign labels with existing X_memmap.npy.
        Assumes X was written in the same traversal order as prepare_training_data():
        for each action in self.actions order, ids ascending.
        """
        X_path = os.path.join(self.mp_data_path, "X_memmap.npy")
        y_path = os.path.join(self.mp_data_path, "y_memmap.npy")

        if not os.path.exists(X_path):
            log("X_memmap.npy missing. You must run 3.Prepare from scratch.")
            return

        # Load X just to get total rows
        X = np.load(X_path, mmap_mode="r")
        total_expected = int(X.shape[0])
        n_classes = len(self.actions)
        if n_classes == 0:
            self.detect_actions_from_folders()
            n_classes = len(self.actions)
            if n_classes == 0:
                log("No actions found. Aborting rebuild."); return

        # Build label map and count how many rows we SHOULD write
        label_map = {label: i for i, label in enumerate(self.actions)}
        log(f"Rebuilding labels for {n_classes} classes...")
        log(f"Label map sample: {list(label_map.items())[:10]} ...")

        # Dry-run to compute target length from discovered ids
        target_len = 0
        per_action_counts = {}
        for action in self.actions:
            ids = self._discover_ids_for_action(action)
            per_action_counts[action] = len(ids)
            target_len += len(ids)

        log(f"X rows: {total_expected} | sum(ids per action): {target_len}")
        if target_len != total_expected:
            log("WARNING: Length mismatch between X rows and discovered ids. "
                "Labels may misalign. Consider full Prepare.")
            # You can choose to abort here if you want strict safety:
            # return

        # Allocate fresh Y and write labels in the same traversal order
        y_mm = np.lib.format.open_memmap(
            y_path, dtype=np.int32, mode='w+', shape=(total_expected,)
        )
        y_mm[:] = -1

        idx = 0
        for action in self.actions:
            lbl = label_map[action]
            for _ in range(per_action_counts[action]):
                if idx >= total_expected:
                    break
                y_mm[idx] = lbl
                idx += 1

        log(f"Rebuilt y_memmap: wrote {idx} labels (of {total_expected}).")
        if np.any(y_mm == -1):
            leftovers = int(np.sum(y_mm == -1))
            log(f"WARNING: {leftovers} rows left as -1. X/Y misalignment likely.")

        # Quick distribution check
        vals, cnts = np.unique(y_mm, return_counts=True)
        log(f"Labels present after rebuild: {vals.tolist()}")
        log(f"Counts per label (first 10): {cnts.tolist()[:10]}")
        log("Done rebuilding labels. Now retrain (5.Train).")
        return y_path

    
        

def main():
    DATA_PATH = r"C:\FSL Model\Filipino_Sign_Data_Complete"
    MP_DATA_PATH = r"C:\FSL Model\MP_DATA"
    processor = SignLanguageProcessor(DATA_PATH, MP_DATA_PATH) 
    X_path = y_cat = train_idx = test_idx = None

    while True:
        print("\n1.Setup  2.Process  3.Prepare  4.Build  5.Train  6.Eval  7.Save  8.Load  9.Realtime  10.Test 11.Debug 12.RebuildY 0.Exit")
        c = input("Choice: ").strip()
        if c == '0':
            break
        elif c == '1':
            processor.setup_data_folders()
        elif c == '2':
            processor.process_videos()
        elif c == '3':
            X_path, y_cat, train_idx, test_idx = processor.prepare_training_data()
        elif c == '4':
            processor.build_model()
        elif c == '5':
            if processor.model and X_path:
                try:
                    e_raw = input("Enter number of epochs (default: 200): ").strip()
                    epochs = int(e_raw) if e_raw.isdigit() else 200
                except Exception:
                    epochs = 200
                processor.train_model(X_path, y_cat, train_idx, test_idx, epochs=epochs)
            else:
                log("Run 3 (Prepare) and 4 (Build) first")
        elif c == '6':
            if processor.model and X_path:
                processor.evaluate_model(X_path, y_cat, test_idx)
            else:
                log("Train first")
        elif c == '7':
            name = input("Enter model name (default: best_fsl_model.h5): ").strip()
            name = name if name else 'best_fsl_model.h5'
            processor.save_model(name)
        elif c == '8':
            name = input("Enter model name (default: best_fsl_model.h5): ").strip()
            name = name if name else 'best_fsl_model.h5'
            processor.load_model(name)
        elif c == '9':
            processor.run_realtime()

        elif c == '10':
            # --- Offline single-sample test (no webcam) ---
            if processor.model is None:
                log("Load or train a model first (8 or 5).")
                continue

            if not processor.actions:
                processor.detect_actions_from_folders()
                if not processor.actions:
                    log("No actions found. Run 1.Setup or 2.Process → 3.Prepare first.")
                    continue

            act = input("Enter action name (folder): ").strip()
            if not act:
                log("No action entered."); continue
            if act not in processor.actions:
                log(f"Action '{act}' not in detected actions. Found: {len(processor.actions)} actions.")
                continue

            ids = processor._discover_ids_for_action(act)
            if not ids:
                log(f"No sample ids found for action '{act}'. Have you run 2.Process/3.Prepare?")
                continue

            vid_raw = input(f"Enter vid id (default {ids[0]}). Available count: {len(ids)}: ").strip()
            try:
                vid_id = int(vid_raw) if vid_raw else ids[0]
            except:
                vid_id = ids[0]

            processor.predict_one(act, vid_id)

        elif c == '11':
            processor.dbg_check_model_vs_actions()
            processor.dbg_label_distribution()
            processor.dbg_data_health()

        elif c == '12':
            # Rebuild labels only (keep X)
            if not processor.actions:
                processor.detect_actions_from_folders()
            processor.rebuild_labels_only()
            # Verify distribution
            processor.dbg_label_distribution()

            processor.predict_one(act, vid_id)
if __name__ == "__main__":
    main()
