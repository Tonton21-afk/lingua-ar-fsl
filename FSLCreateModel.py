import os, cv2, time, math, random, re
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import (
    TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical, Sequence
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# -------------------- utils --------------------

def natural_key(s: str):
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

# -------------------- feature extractors --------------------

def _extract_raw_segments(results, include_face: bool):
    """Return concatenated raw absolute landmarks as a single flat vector:
       pose(33*4) + [face(468*3)] + lh(21*3) + rh(21*3). No normalization."""
    # pose (x,y,z,visibility)
    pose = (np.array([[lm.x, lm.y, lm.z, lm.visibility]
                      for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
            if results.pose_landmarks else np.zeros(33*4, dtype=np.float32))
    # face (x,y,z) optional
    face = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.face_landmarks.landmark], dtype=np.float32).flatten()
            if (include_face and results.face_landmarks) else
            (np.zeros(468*3, dtype=np.float32) if include_face else np.zeros(0, dtype=np.float32)))
    # hands (x,y,z)
    lh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
          if results.left_hand_landmarks else np.zeros(21*3, dtype=np.float32))
    rh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
          if results.right_hand_landmarks else np.zeros(21*3, dtype=np.float32))
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)

def _relative_wrist_to_shoulders(results):
    """6-dim: [Lwrist - Lshoulder (x,y,z), Rwrist - Rshoulder (x,y,z)]"""
    if not results.pose_landmarks:
        return np.zeros(6, dtype=np.float32)
    ls = results.pose_landmarks.landmark[11]  # left shoulder
    rs = results.pose_landmarks.landmark[12]  # right shoulder
    ls_xyz = np.array([ls.x, ls.y, ls.z], dtype=np.float32)
    rs_xyz = np.array([rs.x, rs.y, rs.z], dtype=np.float32)
    # left wrist
    if results.left_hand_landmarks:
        lw = results.left_hand_landmarks.landmark[0]
        lw_xyz = np.array([lw.x, lw.y, lw.z], dtype=np.float32) - ls_xyz
    else:
        lw_xyz = np.zeros(3, dtype=np.float32)
    # right wrist
    if results.right_hand_landmarks:
        rw = results.right_hand_landmarks.landmark[0]
        rw_xyz = np.array([rw.x, rw.y, rw.z], dtype=np.float32) - rs_xyz
    else:
        rw_xyz = np.zeros(3, dtype=np.float32)
    return np.concatenate([lw_xyz, rw_xyz]).astype(np.float32)

# -------------------- main class --------------------

class SignLanguageProcessor:
    def __init__(self, data_path, mp_data_path, include_face: bool = True):
        self.data_path = data_path
        self.mp_data_path = mp_data_path
        self.include_face = include_face

        self.actions = []
        self.model = None
        self.sequence_length = 30

        # smoothing for realtime
        self.smoothed_probs = None
        self.smoothing_factor = 0.7

        # dataset-level normalization stats
        self.norm_mean = None
        self.norm_std = None

    # -------- feature dimensions --------

    def base_feature_len(self):
        # pose 33*4 + hands 2*(21*3) + optional face 468*3
        return (33*4) + (21*3) + (21*3) + ((468*3) if self.include_face else 0)

    def feature_dim(self):
        # base + 6 relative wrist-shoulder deltas
        return self.base_feature_len() + 6

    # -------- admin --------

    def detect_actions_from_folders(self):
        actions = [d for d in os.listdir(self.data_path)
                   if os.path.isdir(os.path.join(self.data_path, d))]
        actions = sorted(actions, key=natural_key)
        self.actions = actions
        log(f"Detected actions ({len(actions)}): {actions}")
        return actions

    def setup_data_folders(self):
        os.makedirs(self.mp_data_path, exist_ok=True)
        self.actions = self.detect_actions_from_folders()
        for action in self.actions:
            for pos in range(self.sequence_length):
                os.makedirs(os.path.join(self.mp_data_path, action, str(pos)), exist_ok=True)
        log(f"Setup completed for {len(self.actions)} actions")
        return True

    # -------- feature vector per frame --------

    def get_feature_vector(self, results):
        """Return raw absolute landmarks + 6 relative deltas (NO normalization here)."""
        base = _extract_raw_segments(results, self.include_face)  # length = base_feature_len
        rel6 = _relative_wrist_to_shoulders(results)              # length = 6
        return np.concatenate([base, rel6]).astype(np.float32)    # total = feature_dim

    # -------- augmentation helpers --------

    def _spatial_shift_vector(self, vec, dx, dy, dz):
        """Shift only absolute coordinates (x,y,z), not pose visibility, and NOT the last 6 relative features."""
        out = vec.copy()
        base_len = self.base_feature_len()
        total_len = self.feature_dim()
        assert total_len == vec.size, "Feature vector length mismatch."

        # Index ranges
        # Pose first: 33*4 (x,y,z,vis)
        off = 0
        for i in range(33):
            b = off + i*4
            out[b+0] += dx
            out[b+1] += dy
            out[b+2] += dz
        off += 33*4

        # Face next (optional): 468*3
        if self.include_face:
            for i in range(468):
                b = off + i*3
                out[b+0] += dx
                out[b+1] += dy
                out[b+2] += dz
            off += 468*3

        # Left hand: 21*3
        for i in range(21):
            b = off + i*3
            out[b+0] += dx
            out[b+1] += dy
            out[b+2] += dz
        off += 21*3

        # Right hand: 21*3
        for i in range(21):
            b = off + i*3
            out[b+0] += dx
            out[b+1] += dy
            out[b+2] += dz
        off += 21*3

        # Sanity: off should now equal base_len; the last 6 dims are relatives -> DO NOT shift
        # (They remain valid because a global translation cancels out in (wrist - shoulder))
        assert off == base_len
        return out

    def _discover_ids_for_action(self, action):
        frame0 = os.path.join(self.mp_data_path, action, "0")
        if not os.path.isdir(frame0):
            return []
        ids = []
        for fname in os.listdir(frame0):
            if not fname.lower().endswith(".npy"):
                continue
            stem, _ = os.path.splitext(fname)
            # original IDs are integers; augmented have suffix
            base = stem.split('_')[0]
            try:
                bid = int(base)
                if bid not in ids:
                    ids.append(bid)
            except ValueError:
                pass
        ids.sort()
        if len(ids) > 1000:
            ids = ids[:1000]
        return ids

    # -------- processing videos --------

    def process_videos(self):
        """Extract and SAVE UNNORMALIZED per-frame feature vectors (absolute + relative)."""
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
                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))],
                key=natural_key
            )
            log(f"Processing {len(video_files)} videos for {action}")

            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as holistic:

                for idx, vf in enumerate(video_files):
                    cap_path = os.path.join(video_dir, vf)
                    cap = cv2.VideoCapture(cap_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    log(f"{action}/{vf} | fps={fps:.1f}, frames={total_frames}")
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

                        vec = self.get_feature_vector(results)  # UNNORMALIZED
                        np.save(os.path.join(data_dir, str(pos), f"{idx}.npy"), vec)

                    cap.release()

                    if (idx + 1) % 100 == 0:
                        log(f"{action}: processed {idx+1}/{len(video_files)} videos")

        log("Finished processing all videos!")
        return True

    def add_data_augmentation(self):
        """Augment saved frames on disk (noise, temporal shift, spatial shift)."""
        log("Starting data augmentation...")
        aug_techniques = ['noise', 'temporal_shift', 'spatial_shift']

        for action in self.actions:
            ids = self._discover_ids_for_action(action)
            log(f"Augmenting {len(ids)} samples for action: {action}")

            for vid_id in ids:
                for aug_type in aug_techniques:
                    self._create_augmented_sequence(action, vid_id, aug_type)

        log("Data augmentation completed!")

    def _create_augmented_sequence(self, action, original_id, aug_type):
        new_id = f"{original_id}_{aug_type}"
        T = self.sequence_length
        F = self.feature_dim()

        if aug_type == 'temporal_shift':
            self._apply_temporal_shift(action, original_id, new_id)
            return

        for pos in range(T):
            orig_path = os.path.join(self.mp_data_path, action, str(pos), f"{original_id}.npy")
            aug_path  = os.path.join(self.mp_data_path, action, str(pos), f"{new_id}.npy")
            if not os.path.exists(orig_path):
                continue

            vec = np.load(orig_path).astype(np.float32, copy=False)
            if vec.size != F:
                # pad/trim defensively
                fixed = np.zeros(F, dtype=np.float32)
                m = min(F, vec.size)
                fixed[:m] = vec[:m]
                vec = fixed

            if aug_type == 'noise':
                noise = np.random.normal(0, 0.01, vec.shape).astype(np.float32)
                # do NOT add noise to last 6 relative features too aggressively; keep them cleaner
                noise[-6:] *= 0.2
                augmented = vec + noise

            elif aug_type == 'spatial_shift':
                dx, dy, dz = np.random.uniform(-0.03, 0.03, 3).astype(np.float32)
                augmented = self._spatial_shift_vector(vec, dx, dy, dz)

            else:
                augmented = vec

            np.save(aug_path, augmented)

    def _apply_temporal_shift(self, action, original_id, new_id):
        shift = random.randint(-3, 3)  # -3..+3
        T = self.sequence_length
        for pos in range(T):
            orig_path = os.path.join(self.mp_data_path, action, str(pos), f"{original_id}.npy")
            aug_path  = os.path.join(self.mp_data_path, action, str(pos), f"{new_id}.npy")
            if not os.path.exists(orig_path):
                continue
            if shift >= 0:
                source_pos = max(0, pos - shift)
            else:
                source_pos = min(T - 1, pos - shift)
            src = os.path.join(self.mp_data_path, action, str(source_pos), f"{original_id}.npy")
            if os.path.exists(src):
                np.save(aug_path, np.load(src))
            else:
                np.save(aug_path, np.load(orig_path))

    def enhanced_process_videos(self):
        self.process_videos()
        self.add_data_augmentation()
        return True
    


    # -------- dataset prep & normalization --------

    def _load_norm_stats(self):
        mpath = os.path.join(self.mp_data_path, "norm_mean.npy")
        spath = os.path.join(self.mp_data_path, "norm_std.npy")
        if os.path.exists(mpath) and os.path.exists(spath):
            self.norm_mean = np.load(mpath)
            self.norm_std  = np.load(spath)
            return True
        return False

    def _save_norm_stats(self, mean, std):
        np.save(os.path.join(self.mp_data_path, "norm_mean.npy"), mean.astype(np.float32))
        np.save(os.path.join(self.mp_data_path, "norm_std.npy"),  std.astype(np.float32))

    def _normalize_batch_inplace(self, batch):
        """
        Normalize non-zero entries per feature using dataset-level stats.
        batch shape: (B, T, F); norm_mean/std shape: (F,)
        """
        if self.norm_mean is None or self.norm_std is None:
            return batch

        # Broadcast (F,) -> (1,1,F)
        mean = self.norm_mean.reshape(1, 1, -1).astype(np.float32)
        std  = self.norm_std.reshape(1, 1, -1).astype(np.float32)

        # Remember where zeros are (padding/missing landmarks)
        nz_mask = batch != 0

        # Normalize everything, then put zeros back
        batch[:] = (batch - mean) / std
        batch[~nz_mask] = 0.0
        return batch


    def compute_dataset_norm_stats(self, X_path):
        """Compute mean/std over non-zero entries per feature from memmap X."""
        X = np.load(X_path, mmap_mode='r')  # shape: [N, T, F]
        N, T, F = X.shape
        
        # Reshape to 2D for easier computation
        X_flat = X.reshape(-1, F)  # [N*T, F]
        
        # Create mask for non-zero elements
        mask = (X_flat != 0)
        
        # Compute mean and std per feature, considering only non-zero elements
        mean = np.zeros(F, dtype=np.float32)
        std = np.zeros(F, dtype=np.float32)
        
        for f in range(F):
            feature_data = X_flat[:, f]
            feature_mask = mask[:, f]
            if np.any(feature_mask):
                non_zero_data = feature_data[feature_mask]
                mean[f] = np.mean(non_zero_data)
                std[f] = np.std(non_zero_data) + 1e-8
            else:
                mean[f] = 0.0
                std[f] = 1.0  # Avoid division by zero
        
        self._save_norm_stats(mean, std)
        self.norm_mean, self.norm_std = mean, std
        log(f"Normalization stats - Mean range: [{mean.min():.3f}, {mean.max():.3f}], Std range: [{std.min():.3f}, {std.max():.3f}]")
        return mean, std

    def prepare_training_data(self):
        if len(self.actions) == 0:
            self.detect_actions_from_folders()
            if len(self.actions) == 0:
                log("No actions found for training!")
                return None, None, None, None

        self.actions = sorted(self.actions, key=natural_key)
        order_path = os.path.join(self.mp_data_path, "actions_order.txt")
        os.makedirs(self.mp_data_path, exist_ok=True)
        with open(order_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.actions))

        label_map = {label: i for i, label in enumerate(self.actions)}
        T = int(self.sequence_length)
        F = int(self.feature_dim())

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
        sentinel = np.iinfo(np.uint16).max

        # Reuse check
        try:
            if os.path.exists(X_path) and os.path.exists(y_path):
                X_try = np.load(X_path, mmap_mode="r")
                y_try = np.load(y_path, mmap_mode="r")
                if (X_try.shape == (total_expected, T, F)
                    and y_try.shape == (total_expected,)
                    and np.all(y_try != sentinel)):
                    log("Dataset already prepared — skipping Step 3 (using existing memmaps).")
                    self._load_norm_stats() or self.compute_dataset_norm_stats(X_path)
                    y_cat = to_categorical(y_try, num_classes=len(self.actions))
                    idxs = np.arange(total_expected)
                    train_idx, test_idx = train_test_split(
                        idxs, test_size=0.05, random_state=42, stratify=y_try
                    )
                    return X_path, y_cat, train_idx, test_idx
                else:
                    log("Memmaps exist but fail integrity checks — rebuilding.")
        except Exception as e:
            log(f"Could not open existing memmaps ({e}) — rebuilding.")

        # fresh build
        for p in (X_path, y_path):
            if os.path.isdir(p):
                raise RuntimeError(f"Expected file but found directory: {p}")
            if os.path.exists(p):
                os.remove(p)

        X_mm = np.lib.format.open_memmap(
            X_path, dtype=np.float32, mode="w+",
            shape=(total_expected, T, F)
        )
        y_mm = np.lib.format.open_memmap(
            y_path, dtype=np.uint16, mode="w+",
            shape=(total_expected,)
        )
        y_mm[:] = sentinel

        idx = 0
        zeros = np.zeros(F, dtype=np.float32)

        def _pad_or_fix(arr):
            if arr.size == F:
                return arr
            out = zeros.copy()
            m = min(F, arr.size)
            out[:m] = arr[:m]
            return out

        for action, ids in action_to_ids.items():
            if not ids:
                continue
            lbl = label_map[action]
            log(f"Preparing data for {len(ids)} videos of action: {action}")
            for i, vid in enumerate(ids):
                win = np.empty((T, F), dtype=np.float32)
                for pos in range(T):
                    npy_path = os.path.join(self.mp_data_path, action, str(pos), f"{vid}.npy")
                    try:
                        arr = np.load(npy_path, allow_pickle=False).astype(np.float32, copy=False)
                        arr = _pad_or_fix(arr)
                    except Exception:
                        arr = zeros
                    win[pos] = arr
                X_mm[idx] = win
                y_mm[idx] = lbl
                idx += 1
                if (i + 1) % 100 == 0 or (i + 1) == len(ids):
                    log(f"  Processed {i+1}/{len(ids)} videos")

        bad = int(np.sum(y_mm == sentinel))
        if bad > 0:
            log(f"ERROR: {bad} labels unfilled in y_memmap (sentinel={sentinel}).")
            raise RuntimeError(f"{bad} labels unfilled in y_memmap. Please re-run Prepare.")

        # compute & save dataset-level normalization stats (on raw X)
        self.compute_dataset_norm_stats(X_path)

        y_cat = to_categorical(y_mm, num_classes=len(self.actions))
        idxs = np.arange(total_expected)
        train_idx, test_idx = train_test_split(
            idxs, test_size=0.05, random_state=42, stratify=y_mm
        )

        log(f"Prepared {total_expected} sequences across {len(self.actions)} actions")
        return X_path, y_cat, train_idx, test_idx

    # -------- model --------

    def build_enhanced_model(self):
        if len(self.actions) == 0:
            self.detect_actions_from_folders()
            if len(self.actions) == 0:
                log("No actions found for model building!")
                return None

        F = self.feature_dim()
        log(f"Building enhanced model for {len(self.actions)} actions, F={F}")

        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.sequence_length, F)),
            BatchNormalization(),
            Dropout(0.3),

            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),

            Bidirectional(LSTM(64, return_sequences=False)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),

            Dense(128, activation='relu'),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.2),

            Dense(len(self.actions), activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy', 'top_k_categorical_accuracy']
        )

        self.model = model
        log("✅ Enhanced model built successfully")
        return model

    def build_model(self):
        return self.build_enhanced_model()

    # -------- training --------

    def enhanced_train_model(self, X_path, y_cat, train_idx, test_idx, epochs=300, batch_size=32):
        X_mem = np.load(X_path, mmap_mode='r')
        y_cat = np.asarray(y_cat)
        seq_len  = X_mem.shape[1]
        feat_dim = X_mem.shape[2]

        # make sure norm stats are loaded
        if (self.norm_mean is None) or (self.norm_std is None):
            self._load_norm_stats() or self.compute_dataset_norm_stats(X_path)

        class DiskSequence(Sequence):
            def __init__(self, indices, batch_size, seq_len, feat_dim, parent):
                self.indices = np.asarray(indices, dtype=np.int64)
                self.batch_size = int(batch_size)
                self.seq_len = int(seq_len)
                self.feat_dim = int(feat_dim)
                self.parent = parent  # to access norm stats
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

                # normalize in-place using dataset stats
                self.parent._normalize_batch_inplace(bx)
                by = y_cat[batch_inds]
                return bx, by

            def on_epoch_end(self):
                np.random.shuffle(self.indices)

        train_seq = DiskSequence(train_idx, batch_size, seq_len, feat_dim, self)
        val_seq   = DiskSequence(test_idx,  batch_size, seq_len, feat_dim, self)

        # class weights
        y_labels = np.argmax(y_cat[train_idx], axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
        class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.mp_data_path, 'logs', run_stamp)
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
            ModelCheckpoint(
                os.path.join(self.mp_data_path, 'enhanced_best_model.h5'),
                monitor='val_categorical_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            TensorBoard(log_dir=log_dir),
        ]

        log(f"Enhanced training started (epochs={epochs}, batch_size={batch_size}) — logs at {log_dir}")
        hist = self.model.fit(
            train_seq,
            epochs=epochs,
            validation_data=val_seq,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
            workers=0,
            use_multiprocessing=False,
            max_queue_size=10
        )
        log("Enhanced training complete")
        return hist

    def train_model(self, X_path, y_cat, train_idx, test_idx, epochs=200, batch_size=32):
        return self.enhanced_train_model(X_path, y_cat, train_idx, test_idx, epochs, batch_size)

    # -------- evaluation --------

    def evaluate_model(self, X_path, y_cat, test_idx, batch_size=32):
        X = np.load(X_path, mmap_mode='r')
        if (self.norm_mean is None) or (self.norm_std is None):
            self._load_norm_stats() or self.compute_dataset_norm_stats(X_path)

        y_true, y_pred = [], []
        for start in range(0, len(test_idx), batch_size):
            end = min(start + batch_size, len(test_idx))
            batch_idx = test_idx[start:end]
            X_batch = X[batch_idx].astype(np.float32, copy=True)
            self._normalize_batch_inplace(X_batch)

            preds = self.model.predict(X_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(y_cat[batch_idx], axis=1))

        acc = accuracy_score(y_true, y_pred)
        labels = list(range(len(self.actions)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(
            y_true, y_pred, labels=labels, target_names=self.actions,
            zero_division=0, output_dict=True
        )

        log(f"Accuracy: {acc*100:.2f}%")
        log(f"Confusion matrix shape: {cm.shape}")
        log("Classification Report:")
        log(f"  Precision (macro): {report['macro avg']['precision']:.3f}")
        log(f"  Recall (macro): {report['macro avg']['recall']:.3f}")
        log(f"  F1-score (macro): {report['macro avg']['f1-score']:.3f}")

        return acc, cm, report

    # -------- save/load --------

    def save_model(self, model_name='enhanced_sign_language_model.h5'):
        path = os.path.join(self.mp_data_path, model_name)
        self.model.save(path)
        log(f"Enhanced model saved at {path}")

    def load_model(self, model_name='enhanced_sign_language_model.h5'):
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

        out_units = int(self.model.output_shape[-1])
        if out_units != len(self.actions):
            log(f"ERROR: model outputs {out_units} classes but actions has {len(self.actions)}. Rebuild with consistent order.")
        # load normalization stats if present
        self._load_norm_stats()
        log(f"Enhanced model loaded from {path}")
        return self.model

    # -------- realtime --------

    def enhanced_run_realtime(self, **kwargs):
        self.smoothed_probs = None
        kwargs['use_enhanced_features'] = True  # retained arg for compatibility
        return self.run_realtime(**kwargs)

    def run_realtime(
        self,
        model_path=None,
        prob_threshold: float = 0.60,
        show_fps: bool = True,
        camera_index: int = 0,
        frame_size: tuple = (960, 540),
        hist_k: int = 7,
        vote_k: int = 4,
        min_det_conf: float = 0.50,
        min_track_conf: float = 0.50,
        log_interval: float = 0.50,
        softmax_safety: bool = False,
        use_enhanced_features: bool = True  # kept for API compatibility (ignored; we now always use get_feature_vector)
    ):
        if self.model is None:
            if model_path is not None:
                try:
                    self.model = load_model(model_path)
                    log(f"Loaded model for realtime from: {model_path}")
                except Exception as e:
                    log(f"Failed to load model at {model_path}: {e}")
                    return
            else:
                log("No model in memory. Load/train first or pass model_path.")
                return

        if not self.actions:
            self.detect_actions_from_folders()
            if not self.actions:
                log("No actions found. Cannot label predictions.")
                return

        # load normalization stats
        if (self.norm_mean is None) or (self.norm_std is None):
            if not self._load_norm_stats():
                log("WARN: normalization stats not found; realtime will use unnormalized vectors.")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            log(f"Cannot open webcam index {camera_index}. Try camera_index=1.")
            return
        if frame_size and len(frame_size) == 2:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

        window   = deque(maxlen=self.sequence_length)
        pred_hist = deque(maxlen=hist_k)
        confidence_hist = deque(maxlen=hist_k)

        prev_time = time.time()
        fps_hist  = deque(maxlen=20)
        fps = 0.0
        thr = float(prob_threshold)
        last_log_t = 0.0
        mirror = False

        try:
            with mp_holistic.Holistic(
                min_detection_confidence=min_det_conf,
                min_tracking_confidence=min_track_conf
            ) as holistic:

                log("Enhanced realtime started. Keys: 'q' quit | 't' thr-0.05 | 'y' thr+0.05 | 'm' mirror on/off")

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frame_size:
                        frame = cv2.resize(frame, frame_size)

                    if mirror:
                        frame = cv2.flip(frame, 1)

                    image, results = mediapipe_detection(frame, holistic)
                    vec = self.get_feature_vector(results)  # raw (abs + relative)

                    window.append(vec)

                    # Draw landmarks
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    det_count = 0
                    if results.pose_landmarks:        det_count += 1
                    if results.left_hand_landmarks:   det_count += 1
                    if results.right_hand_landmarks:  det_count += 1

                    if show_fps:
                        now = time.time()
                        dt = now - prev_time
                        prev_time = now
                        if dt > 0:
                            fps_hist.append(1.0 / dt)
                            fps = sum(fps_hist) / max(1, len(fps_hist))
                        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if len(window) < self.sequence_length:
                        cv2.putText(image, f"Collecting frames... {len(window)}/{self.sequence_length}",
                                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.putText(image, f"LM sets: {det_count}/3",
                                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow("FSL Enhanced Realtime Recognition", image)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'): break
                        elif key == ord('m'): mirror = not mirror
                        continue

                    # build batch (1, T, F) and normalize
                    seq = np.expand_dims(np.array(window, dtype=np.float32), axis=0)
                    self._normalize_batch_inplace(seq)

                    probs = self.model.predict(seq, verbose=0)[0]

                    # exponential smoothing
                    if self.smoothed_probs is None:
                        self.smoothed_probs = probs
                    else:
                        self.smoothed_probs = (self.smoothing_factor * self.smoothed_probs +
                                               (1 - self.smoothing_factor) * probs)

                    final_probs = self.smoothed_probs

                    if softmax_safety:
                        if (final_probs.ndim == 1) and (np.any(final_probs < 0) or np.max(final_probs) > 1.0 or np.sum(final_probs) <= 0.0):
                            ex = np.exp(final_probs - np.max(final_probs))
                            denom = np.clip(np.sum(ex), 1e-8, None)
                            final_probs = ex / denom

                    if not np.isfinite(final_probs).all():
                        log("WARN: non-finite probabilities detected; sanitizing.")
                        final_probs = np.nan_to_num(final_probs, nan=0.0, posinf=0.0, neginf=0.0)

                    num_classes = len(self.actions)
                    if final_probs.shape[0] != num_classes:
                        final_probs = final_probs[:num_classes] if final_probs.shape[0] > num_classes \
                            else np.concatenate([final_probs, np.zeros((num_classes - final_probs.shape[0],), final_probs.dtype)], 0)

                    order = np.argsort(final_probs)[::-1]
                    topk = order[:3]
                    topk_probs = final_probs[topk]

                    top_idx = int(topk[0])
                    confidence = float(final_probs[top_idx])

                    pred_hist.append(top_idx)
                    confidence_hist.append(confidence)

                    avg_confidence = float(np.mean(confidence_hist)) if confidence_hist else confidence
                    dynamic_threshold = max(0.5, thr * (0.8 + 0.2 * avg_confidence))

                    if len(pred_hist) > 0:
                        stable = max(set(pred_hist), key=pred_hist.count)
                        stable_votes = pred_hist.count(stable)
                        stable_confidence = float(final_probs[stable])
                    else:
                        stable, stable_votes, stable_confidence = top_idx, 1, confidence

                    if (stable_votes >= vote_k and stable_confidence >= dynamic_threshold and
                        avg_confidence >= thr):
                        final_idx = int(stable)
                        final_conf = stable_confidence
                    else:
                        final_idx = int(top_idx)
                        final_conf = confidence

                    final_label = self.actions[final_idx] if final_conf >= thr else "Unsure"

                    # overlay
                    h, w = image.shape[:2]
                    left, base_y, line_h = 20, h - 120, 28

                    def draw_text_with_bg(img, text, pos, font, scale, color, thickness, bg=(0,0,0), alpha=0.5):
                        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                        x, y = pos
                        x1, y1 = x-6, y-th-6
                        x2, y2 = x+tw+6, y+6
                        overlay = img.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
                        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
                        cv2.putText(img, text, (x, y), font, scale, color, thickness)

                    color = (0, 255, 0) if final_conf >= 0.8 else (0, 255, 255) if final_conf >= 0.6 else (0, 165, 255)

                    draw_text_with_bg(image, f"{final_label}", (left, base_y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    for i, (cls_idx, p) in enumerate(zip(topk, topk_probs), start=1):
                        draw_text_with_bg(image, f"{i}. {self.actions[int(cls_idx)]}: {p*100:.1f}%",
                                          (left, base_y + i*line_h),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    draw_text_with_bg(image, f"thr={thr:.2f} (dyn={dynamic_threshold:.2f})", (w - 250, 40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    draw_text_with_bg(image, f"LM sets: {det_count}/3", (w - 220, 70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    draw_text_with_bg(image, f"Conf: {final_conf:.2f}", (w - 200, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    now_t = time.time()
                    if now_t - last_log_t >= log_interval:
                        last_log_t = now_t
                        top5 = order[:5]
                        log(f"ENHANCED TOPK: {[(self.actions[int(i)], round(float(final_probs[i]),4)) for i in top5]} "
                            f"| final=({final_label}, {final_conf:.3f}) dyn_thr={dynamic_threshold:.2f}")

                    cv2.imshow("FSL Enhanced Realtime Recognition", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('t'):
                        thr = max(0.05, round(thr - 0.05, 2))
                    elif key == ord('y'):
                        thr = min(0.95, round(thr + 0.05, 2))
                    elif key == ord('m'):
                        mirror = not mirror

        finally:
            cap.release()
            cv2.destroyAllWindows()
            log("Enhanced realtime recognition ended.")

    # -------- test/predict & debug --------

    def predict_one(self, action: str, vid_id: int):
        if self.model is None:
            log("Load/train a model first (option 8 or 5)."); return
        if not self.actions:
            self.detect_actions_from_folders()
        out_units = int(self.model.output_shape[-1])
        if out_units != len(self.actions):
            log(f"ERROR: model outputs {out_units} classes but actions has {len(self.actions)}. "
                f"Check actions_order.txt alignment."); return

        T, F = self.sequence_length, self.feature_dim()
        seq = []
        for pos in range(T):
            path = os.path.join(self.mp_data_path, action, str(pos), f"{vid_id}.npy")
            if not os.path.exists(path):
                log(f"Missing frame file: {path}"); return
            arr = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            if arr.size != F:
                fixed = np.zeros(F, np.float32)
                fixed[:min(F, arr.size)] = arr[:min(F, arr.size)]
                arr = fixed
            seq.append(arr)
        seq = np.expand_dims(np.stack(seq, axis=0), axis=0)

        # normalize
        self._load_norm_stats()
        self._normalize_batch_inplace(seq)

        probs = self.model.predict(seq, verbose=0)[0]
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(probs)[::-1]
        top5 = order[:5]
        log(f"ONE SAMPLE → TOP5: {[(self.actions[int(i)], float(probs[i])) for i in top5]}")
        top1 = int(top5[0])
        log(f"TOP1: {self.actions[top1]} ({float(probs[top1])*100:.2f}%)")

    def rebuild_labels_only(self):
        X_path = os.path.join(self.mp_data_path, "X_memmap.npy")
        y_path = os.path.join(self.mp_data_path, "y_memmap.npy")

        if not os.path.exists(X_path):
            log("X_memmap.npy missing. You must run 3.Prepare from scratch.")
            return

        if not self.actions:
            self.detect_actions_from_folders()
            if not self.actions:
                log("No actions found. Aborting rebuild."); return

        X = np.load(X_path, mmap_mode="r")
        total_expected = int(X.shape[0])

        label_map = {label: i for i, label in enumerate(self.actions)}
        log(f"Rebuilding labels for {len(self.actions)} classes...")

        target_len = 0
        per_action_counts = {}
        for action in self.actions:
            ids = self._discover_ids_for_action(action)
            per_action_counts[action] = len(ids)
            target_len += len(ids)

        log(f"X rows: {total_expected} | sum(ids per action): {target_len}")
        if target_len != total_expected:
            log("WARNING: Length mismatch between X rows and discovered ids. Consider full Prepare.")

        y_mm = np.lib.format.open_memmap(y_path, dtype=np.uint16, mode='w+', shape=(total_expected,))
        y_mm[:] = np.iinfo(np.uint16).max

        idx = 0
        for action in self.actions:
            lbl = label_map[action]
            for _ in range(per_action_counts[action]):
                if idx >= total_expected:
                    break
                y_mm[idx] = lbl
                idx += 1

        log(f"Rebuilt y_memmap: wrote {idx} labels (of {total_expected}).")
        sentinel = np.iinfo(np.uint16).max
        if np.any(y_mm == sentinel):
            leftovers = int(np.sum(y_mm == sentinel))
            log(f"WARNING: {leftovers} rows left as sentinel ({sentinel}). X/Y misalignment likely.")

        vals, cnts = np.unique(y_mm, return_counts=True)
        log(f"Labels present after rebuild: {vals.tolist()}")
        log(f"Counts per label (first 10): {cnts.tolist()[:10]}")
        log("Done rebuilding labels. Now retrain (5.Train).")
        return y_path

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

    def enhanced_pipeline(self):
        log("Starting enhanced processing pipeline...")
        self.setup_data_folders()
        self.enhanced_process_videos()
        X_path, y_cat, train_idx, test_idx = self.prepare_training_data()
        self.build_enhanced_model()
        self.enhanced_train_model(X_path, y_cat, train_idx, test_idx)
        log("Enhanced pipeline completed successfully!")
        return True
    
    def check_augmentation_coverage(self):
        """Verify augmentation is properly included"""
        log("=== AUGMENTATION CHECK ===")
        
        for action in self.actions[:3]:  # Check first 3 actions
            frame0 = os.path.join(self.mp_data_path, action, "0")
            if not os.path.isdir(frame0):
                continue
                
            files = [f for f in os.listdir(frame0) if f.endswith('.npy')]
            original_files = [f for f in files if not '_' in f]
            augmented_files = [f for f in files if '_' in f]
            
            log(f"{action}: {len(original_files)} originals, {len(augmented_files)} augmented")
            
            # Show augmentation types
            aug_types = set()
            for f in augmented_files:
                aug_type = f.split('_')[-1].split('.')[0]
                aug_types.add(aug_type)
            
            if aug_types:
                log(f"  Augmentation types: {sorted(aug_types)}")


# -------------------- CLI --------------------

def main():
    DATA_PATH = r"C:\FSL Model\Filipino_Sign_Data_Complete"
    MP_DATA_PATH = r"C:\FSL Model\MP_DATA"

    # include_face=True keeps your current 1668-dim setup; set False to shrink features
    processor = SignLanguageProcessor(DATA_PATH, MP_DATA_PATH, include_face=True)

    X_path = y_cat = train_idx = test_idx = None

    while True:
        print("\n1.Setup  2.Process  3.Prepare  4.Build  5.Train  6.Eval  7.Save  8.Load  9.Realtime  10.Test 11.Debug 12.RebuildY 13.EnhancedPipeline 14. augementation check sah 15. Add augmentation 0.Exit")
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
                acc, cm, report = processor.evaluate_model(X_path, y_cat, test_idx)
            else:
                log("Train first")
        elif c == '7':
            name = input("Enter model name (default: enhanced_sign_language_model.h5): ").strip()
            name = name if name else 'enhanced_sign_language_model.h5'
            processor.save_model(name)
        elif c == '8':
            name = input("Enter model name (default: enhanced_sign_language_model.h5): ").strip()
            name = name if name else 'enhanced_sign_language_model.h5'
            processor.load_model(name)
        elif c == '9':
            processor.run_realtime()
        elif c == '10':
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
            if not processor.actions:
                processor.detect_actions_from_folders()
            processor.rebuild_labels_only()
            processor.dbg_label_distribution()
        elif c == '13':
            processor.enhanced_pipeline()
        elif c == '14':
            processor.check_augmentation_coverage()
        # in your menu loop:
        elif c == '15':  # Add Aug
            if not processor.actions:
                processor.detect_actions_from_folders()
            processor.add_data_augmentation()          # <- creates files like 12_noise.npy, etc.
            log("Augmentation done. Press 14 to verify.")

        elif c == '16':  # Integrate Aug -> numeric IDs
            import os
            ACTIONS = [d for d in os.listdir(processor.mp_data_path)
                    if os.path.isdir(os.path.join(processor.mp_data_path, d)) and d != "logs"]
            integrate_augmented_as_numeric(processor.mp_data_path, ACTIONS, T=processor.sequence_length)
            log("Integrated augmented files into numeric IDs. Now run 3→4→5.")


if __name__ == "__main__":
    main()
