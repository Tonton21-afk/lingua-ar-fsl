import os, cv2, time, math, random
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


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
        for action in self.actions:
            video_dir = os.path.join(self.data_path, action)
            data_dir = os.path.join(self.mp_data_path, action)
            if not os.path.isdir(video_dir):
                log(f"Warning: Missing video directory for action '{action}', skipping.")
                continue
            video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4','.avi','.mov','.mkv','.MP4','.webm'))]
            log(f"Processing {len(video_files)} videos for {action}")
            for idx, vf in enumerate(video_files[:1000]):
                cap = cv2.VideoCapture(os.path.join(video_dir, vf))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0:
                    cap.release()
                    continue
                frame_idx = np.linspace(0, total_frames-1, self.sequence_length, dtype=int)
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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

    def _discover_ids_for_action(self, action):
        frame0 = os.path.join(self.mp_data_path, action, '0')
        if not os.path.exists(frame0):
            return []
        ids = []
        for f in os.listdir(frame0):
            if f.endswith('.npy'):
                name, _ = os.path.splitext(f)
                try:
                    ids.append(int(name))
                except ValueError:
                    continue
        ids.sort()
        if len(ids) > 1000:
            ids = ids[:1000]
        return ids

    def prepare_training_data(self):
        if len(self.actions) == 0:
            self.detect_actions_from_folders()
            if len(self.actions) == 0:
                log("No actions found for training!")
                return None, None, None, None
        
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
                try:
                    os.remove(X_path)
                except Exception:
                    pass
                try:
                    os.remove(y_path)
                except Exception:
                    pass
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
        log(f"Model loaded from {path}")
        return self.model
    
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

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            log("Realtime recognition started. Press 'q' to quit.")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                window.append(keypoints)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                label_text = "…"
                if len(window) == self.sequence_length:
                    seq = np.expand_dims(np.array(window, dtype=np.float32), axis=0)
                    probs = self.model.predict(seq, verbose=0)[0]
                    top_idx = int(np.argmax(probs))
                    top_prob = float(probs[top_idx])

                    if top_prob >= prob_threshold:
                        label_text = f"{self.actions[top_idx]}  ({top_prob*100:.1f}%)"
                    else:
                        label_text = f"Unsure ({top_prob*100:.1f}%)"

                if show_fps:
                    now = time.time()
                    dt = now - prev_time
                    prev_time = now
                    fps = 1.0 / dt if dt > 0 else fps
                    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.rectangle(image, (10, 50), (650, 90), (0, 0, 0), -1)
                cv2.putText(image, label_text, (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                cv2.imshow("FSL Realtime Recognition", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        log("Realtime recognition ended.")
        
def main():
    DATA_PATH = r"C:\Users\zhyrex\Desktop\cloning\Filipino_Sign_Data_Complete"
    MP_DATA_PATH = r"C:\Users\zhyrex\Desktop\cloning\MP_Data"
    processor = SignLanguageProcessor(DATA_PATH, MP_DATA_PATH)
    X_path = y_cat = train_idx = test_idx = None

    while True:
        print("\n1.Setup  2.Process  3.Prepare  4.Build  5.Train  6.Eval  7.Save  8.Load  9.Realtime  0.Exit")
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

if __name__ == "__main__":
    main()
