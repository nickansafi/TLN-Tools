#Requirement Library
import os
import glob
from collections import defaultdict
from sklearn.utils import shuffle
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import sqlite3
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.losses import huber
from tensorflow.keras.optimizers import Adam

# Check GPU availability
gpu_available = tf.test.is_gpu_available()
print('GPU AVAILABLE:', gpu_available)
typestore = get_typestore(Stores.ROS2_HUMBLE)
typestore.register(get_types_from_msg(
    "float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\n"
    "float32 acceleration\nfloat32 jerk",
    "ackermann_msgs/msg/AckermannDrive",
))
typestore.register(get_types_from_msg(
    "std_msgs/Header header\nackermann_msgs/AckermannDrive drive",
    "ackermann_msgs/msg/AckermannDriveStamped",
))

#========================================================
# Configuration
#========================================================
RESOLUTION_SCALES = [1.0, 0.75]
CALIBRATE_BN = True
CALIBRATION_MAX_BATCHES = 200
BN_DEBUG = True
MATCH = False
INCREMENTAL = False

TEST_MODE = False  # Flip to True for single-batch pipeline test

DATASET_ROOT = './Dataset'

# Early stopping: stop if val loss doesn't improve for this many epochs
PATIENCE = 5
# Minimum improvement to count as "better"
MIN_DELTA = 1e-5
# Maximum epochs per phase (acts as a safety cap)
MAX_EPOCHS = 200

# Phase 2: these specific files only
PHASE2_FILENAMES = [
    'jfr1.db3',
    'jfr2.db3',
    'jfr5_opp.db3',
    'jfr6_opp.db3',
    'Forza_GLC_smile_PP_edgecases_0.db3',
]

# Find phase 2 files by searching anywhere under Dataset/
ALL_DB3 = sorted(glob.glob(os.path.join(DATASET_ROOT, '**', '*.db3'), recursive=True))

PHASE2_PATHS = []
for db3_path in ALL_DB3:
    if os.path.basename(db3_path) in PHASE2_FILENAMES:
        PHASE2_PATHS.append(db3_path)

# Check that we found all of them
found_names = {os.path.basename(p) for p in PHASE2_PATHS}
missing = set(PHASE2_FILENAMES) - found_names
if missing:
    print(f"WARNING: Phase 2 files not found: {missing}")

# Phase 1: everything else (mutually exclusive)
phase2_abs = {os.path.abspath(p) for p in PHASE2_PATHS}
PHASE1_PATHS = [p for p in ALL_DB3 if os.path.abspath(p) not in phase2_abs]

print(f"Phase 1 datasets ({len(PHASE1_PATHS)}) — excludes Phase 2:")
for p in PHASE1_PATHS:
    print(f"  {p}")
print(f"Phase 2 datasets ({len(PHASE2_PATHS)}):")
for p in PHASE2_PATHS:
    print(f"  {p}")

loss_figure_path = './Figures/loss_curve.png'
down_sample_param = 1
lr = 5e-5
batch_size = 64
warmup_epochs = 2
hz = 40

model_name = 'TLN'
model_files = [
    './Benchmark/f1tenth_benchmarks/zarrar/' + model_name + '_noquantized.tflite',
    './Benchmark/f1tenth_benchmarks/zarrar/' + model_name + '_int8.tflite'
]

#========================================================
# Helpers
#========================================================

def compute_conv_output_length(length, kernel_size, stride):
    return (length - kernel_size) // stride + 1


def get_bn_native_lengths(model, input_length):
    lengths = []
    length = input_length
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            length = compute_conv_output_length(
                length, layer.kernel_size[0], layer.strides[0])
        elif isinstance(layer, ResolutionAwareBatchNormalization):
            lengths.append(length)
    return lengths


def patch_bn_native_lengths(model, input_length):
    bn_native_lengths = get_bn_native_lengths(model, input_length)
    bn_layers = [l for l in model.layers
                 if isinstance(l, ResolutionAwareBatchNormalization)]
    for layer, native_len in zip(bn_layers, bn_native_lengths):
        layer.native_length = native_len
        layer.expected_lengths = [
            int(round(native_len * s)) for s in layer.resolution_scales
        ]
    return bn_native_lengths


def resize_lidar_1d(batch, scale):
    if abs(scale - 1.0) < 1e-6:
        return batch
    if len(batch.shape) == 1:
        original_len = batch.shape[0]
        new_len = int(round(original_len * scale))
        if new_len == original_len:
            return batch
        x_old = np.linspace(0, 1, original_len)
        x_new = np.linspace(0, 1, new_len)
        return np.interp(x_new, x_old, batch).astype(np.float32)

    original_len = batch.shape[1]
    new_len = int(round(original_len * scale))
    if new_len == original_len:
        return batch
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, new_len)
    resized = np.array([np.interp(x_new, x_old, sample) for sample in batch])
    return resized.astype(np.float32)


def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min


def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return np.mean(loss)


#========================================================
# Learning Rate Schedule
#========================================================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then cosine decay to zero."""

    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.base_lr * (step / tf.maximum(warmup, 1.0))
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(np.pi * progress))

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
        }


def build_lr_schedule(base_lr, num_epochs, steps_per_epoch, warmup_epochs=2):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    return WarmupCosineDecay(base_lr, warmup_steps, total_steps)


#========================================================
# Resolution-Aware Batch Normalization
#========================================================

class ResolutionAwareBatchNormalization(tf.keras.layers.Layer):

    def __init__(self, resolution_scales, native_length=1,
                 momentum=0.1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.resolution_scales = [float(s) for s in resolution_scales]
        self.num_scales = len(self.resolution_scales)
        self.native_length = native_length
        self.momentum = momentum
        self.epsilon = epsilon
        self._current_scale_index = 0
        self.expected_lengths = [
            int(round(native_length * s)) for s in self.resolution_scales
        ]

    def build(self, input_shape):
        nf = input_shape[-1]
        self.gamma = self.add_weight(
            'gamma', shape=(nf,), initializer='ones', trainable=True)
        self.beta = self.add_weight(
            'beta', shape=(nf,), initializer='zeros', trainable=True)
        self.all_running_means = self.add_weight(
            'all_running_means', shape=(self.num_scales, nf),
            initializer='zeros', trainable=False)
        self.all_running_vars = self.add_weight(
            'all_running_vars', shape=(self.num_scales, nf),
            initializer='ones', trainable=False)
        self.all_num_batches = self.add_weight(
            'all_num_batches', shape=(self.num_scales,),
            initializer='zeros', dtype=tf.float32, trainable=False)
        super().build(input_shape)

    def set_scale_index(self, index):
        self._current_scale_index = index

    def _resolve_index_from_shape(self, x):
        actual_len = tf.cast(tf.shape(x)[1], tf.float32)
        expected = tf.constant(self.expected_lengths, dtype=tf.float32)
        return tf.argmin(tf.abs(expected - actual_len), output_type=tf.int32)

    def call(self, x, training=False):
        if training:
            idx = self._current_scale_index

            if BN_DEBUG:
                spatial_dim = x.shape[1] if x.shape[1] is not None else '?'
                scale = self.resolution_scales[idx]
                print(
                    f"  [BN] {self.name} | train | bank={idx} "
                    f"(scale={scale}) | spatial={spatial_dim} | "
                    f"expected={self.expected_lengths}"
                )

            mean = tf.reduce_mean(x, axis=[0, 1])
            var = tf.math.reduce_variance(x, axis=[0, 1])

            m = self.momentum
            if m is None:
                count = self.all_num_batches[idx]
                m = 1.0 / (tf.cast(count, tf.float32) + 1.0)

            new_means = self.all_running_means.numpy()
            new_vars = self.all_running_vars.numpy()
            new_counts = self.all_num_batches.numpy()
            new_means[idx] = new_means[idx] * (1.0 - m) + mean.numpy() * m
            new_vars[idx] = new_vars[idx] * (1.0 - m) + var.numpy() * m
            new_counts[idx] += 1.0
            self.all_running_means.assign(new_means)
            self.all_running_vars.assign(new_vars)
            self.all_num_batches.assign(new_counts)
        else:
            idx = self._resolve_index_from_shape(x)

            if BN_DEBUG and tf.executing_eagerly():
                spatial_dim = x.shape[1] if x.shape[1] is not None else '?'
                idx_val = int(idx.numpy())
                scale = self.resolution_scales[idx_val]
                print(
                    f"  [BN] {self.name} | infer | bank={idx_val} "
                    f"(scale={scale}) | spatial={spatial_dim} | "
                    f"expected={self.expected_lengths}"
                )

            mean = tf.gather(self.all_running_means, idx)
            var = tf.gather(self.all_running_vars, idx)

        x_norm = (x - mean) / tf.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta

    def reset_running_stats(self):
        self.all_running_means.assign(tf.zeros_like(self.all_running_means))
        self.all_running_vars.assign(tf.ones_like(self.all_running_vars))
        self.all_num_batches.assign(tf.zeros_like(self.all_num_batches))

    def get_config(self):
        config = super().get_config()
        config.update({
            'resolution_scales': self.resolution_scales,
            'native_length': self.native_length,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
        })
        return config


def set_model_resolution(model, scale_index):
    for layer in model.layers:
        if isinstance(layer, ResolutionAwareBatchNormalization):
            layer.set_scale_index(scale_index)


#========================================================
# BN Calibration
#========================================================

def calibrate_batch_norm(model, lidar_data, resolution_scales,
                         batch_size=64, max_batches=200):
    bn_layers = [l for l in model.layers
                 if isinstance(l, ResolutionAwareBatchNormalization)]
    if not bn_layers:
        return

    original_momenta = []
    for layer in bn_layers:
        original_momenta.append(layer.momentum)
        layer.reset_running_stats()
        layer.momentum = None

    batches_processed = 0
    for start in range(0, len(lidar_data), batch_size):
        batch = lidar_data[start:start + batch_size]
        if len(batch) == 0:
            break
        for scale_idx, scale in enumerate(resolution_scales):
            set_model_resolution(model, scale_idx)
            resized = resize_lidar_1d(batch, scale)
            resized = np.expand_dims(resized, axis=-1).astype(np.float32)
            model(resized, training=True)
        batches_processed += 1
        if max_batches is not None and batches_processed >= max_batches:
            break

    for layer, m in zip(bn_layers, original_momenta):
        layer.momentum = m
    print(f"BN calibration complete: {batches_processed} batches, "
          f"{len(resolution_scales)} scales")


#========================================================
# Dataset Loading
#========================================================

def load_dataset(db3_paths, down_sample_param=1):
    """Load lidar/servo/speed from a list of .db3 files."""
    all_lidar = []
    all_servo = []
    all_speed = []
    max_speed_seen = 0.0

    for pth in db3_paths:
        if not os.path.exists(pth):
            print(f"  WARNING: {pth} not found, skipping")
            continue

        lidar_data = []
        servo_data = []
        speed_data = []

        connection = sqlite3.connect(pth)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT topics.name, topics.type, messages.data "
            "FROM messages JOIN topics ON messages.topic_id = topics.id"
        )
        for topic, msg_type, rawdata in cursor:
            try:
                msg = typestore.deserialize_cdr(rawdata, msg_type)
            except Exception:
                continue
            if topic in ['Lidar', 'scan']:
                if len(msg.ranges) != 1081 and MATCH:
                    continue
                ranges = msg.ranges[:1080][::down_sample_param]
                lidar_data.append(ranges)
            if topic in ['Ackermann', 'drive']:
                servo_data.append(msg.drive.steering_angle)
                s = msg.drive.speed
                speed_data.append(s)
                if s > max_speed_seen:
                    max_speed_seen = s
        connection.close()

        if len(set([len(lidar_data), len(servo_data), len(speed_data)])) != 1:
            print(f"  WARNING: mismatched lengths in {pth}, skipping")
            continue

        all_lidar.extend(lidar_data)
        all_servo.extend(servo_data)
        all_speed.extend(speed_data)
        print(f"  Loaded {len(lidar_data)} samples from {pth}")

    return (
        np.asarray(all_lidar),
        np.asarray(all_servo),
        np.asarray(all_speed),
        max_speed_seen,
    )


def prepare_split(lidar, servo, speed, max_speed, min_speed=0,
                  train_ratio=0.85, seed=62):
    """Shuffle, normalize speed, train/test split."""
    speed_norm = linear_map(speed, min_speed, max_speed, 0, 1)
    labels = np.stack([servo, speed_norm], axis=1)

    lidar_shuf = shuffle(lidar, random_state=seed)
    labels_shuf = shuffle(labels, random_state=seed)

    n_train = int(train_ratio * len(lidar_shuf))
    return {
        'train_lidar': lidar_shuf[:n_train],
        'train_labels': labels_shuf[:n_train],
        'test_lidar': lidar_shuf[n_train:],
        'test_labels': labels_shuf[n_train:],
    }


#========================================================
# Training Loop (one phase) with early stopping
#========================================================

def run_training_phase(model, data, max_epochs, base_lr, batch_size,
                       warmup_epochs, patience, min_delta,
                       phase_name="Phase"):
    """Multi-resolution training with warmup + cosine decay and
    early stopping based on average validation loss.

    Stops when val loss hasn't improved by at least min_delta
    for `patience` consecutive epochs. Restores the best weights.
    """
    global BN_DEBUG

    train_lidar = data['train_lidar']
    train_labels = data['train_labels']
    test_lidar = data['test_lidar']
    test_labels = data['test_labels']

    effective_max = 1 if TEST_MODE else max_epochs

    steps_per_epoch = max(1, len(train_lidar) // batch_size)
    schedule = build_lr_schedule(
        base_lr, effective_max, steps_per_epoch, warmup_epochs)
    optimizer = Adam(learning_rate=schedule)
    loss_fn = tf.keras.losses.Huber()

    history_train = []
    history_val = []
    history_val_per_res = defaultdict(list)

    # Early stopping state
    best_val_loss = float('inf')
    best_weights = None
    epochs_without_improvement = 0

    print(f"\n{'='*60}")
    print(f" {phase_name}: up to {effective_max} epochs, base_lr={base_lr:.2e}")
    print(f" Early stopping: patience={patience}, min_delta={min_delta:.1e}")
    if TEST_MODE:
        print(f" *** TEST MODE: 1 epoch, 1 batch only ***")
    print(f" {len(train_lidar)} train / {len(test_lidar)} test samples")
    print(f" Warmup: {warmup_epochs} epochs ({warmup_epochs * steps_per_epoch} steps)")
    print(f" Steps/epoch: {steps_per_epoch}, "
          f"max total: {effective_max * steps_per_epoch}")
    print(f"{'='*60}\n")

    phase_start = time.time()

    for epoch in range(effective_max):
        epoch_start = time.time()
        perm = np.random.permutation(len(train_lidar))
        lidar_shuf = train_lidar[perm]
        labels_shuf = train_labels[perm]

        epoch_loss = 0.0
        num_batches = 0
        show_debug = BN_DEBUG and epoch == 0

        for start in range(0, len(lidar_shuf) - batch_size + 1, batch_size):
            lidar_batch = lidar_shuf[start:start + batch_size]
            label_batch = labels_shuf[start:start + batch_size]
            label_tensor = tf.constant(label_batch, dtype=tf.float32)

            first_batch = show_debug and num_batches == 0

            with tf.GradientTape() as tape:
                total_loss = 0.0
                for scale_idx, scale in enumerate(RESOLUTION_SCALES):
                    set_model_resolution(model, scale_idx)
                    resized = resize_lidar_1d(lidar_batch, scale)
                    resized = np.expand_dims(resized, -1).astype(np.float32)

                    if first_batch:
                        print(f"  Scale {scale}x (bank {scale_idx}), "
                              f"input shape {resized.shape}:")

                    preds = model(resized, training=True)
                    total_loss += tf.reduce_mean(loss_fn(label_tensor, preds))

                avg_loss = total_loss / len(RESOLUTION_SCALES)

            grads = tape.gradient(avg_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if first_batch:
                BN_DEBUG = False

            epoch_loss += float(avg_loss)
            num_batches += 1

            if num_batches % 100 == 0 or num_batches == 1 and INCREMENTAL:
                current_step = (epoch * steps_per_epoch) + num_batches
                current_lr = float(schedule(current_step))
                print(
                    f'    batch {num_batches}/{steps_per_epoch} '
                    f'loss:{float(avg_loss):.4f} lr:{current_lr:.2e}')

            if TEST_MODE:
                print(f"  [TEST MODE] Trained 1 batch, stopping epoch.")
                break

        avg_epoch_loss = epoch_loss / max(num_batches, 1)

        # Per-resolution validation
        val_losses = {}
        for scale_idx, scale in enumerate(RESOLUTION_SCALES):
            set_model_resolution(model, scale_idx)
            resized_test = resize_lidar_1d(test_lidar, scale)
            resized_test = np.expand_dims(resized_test, -1).astype(np.float32)

            val_preds = []
            for vstart in range(0, len(resized_test), batch_size):
                vbatch = resized_test[vstart:vstart + batch_size]
                vp = model(vbatch, training=False)
                val_preds.append(vp.numpy())
                if TEST_MODE:
                    break
            val_preds = np.concatenate(val_preds, axis=0)
            val_losses[scale] = float(tf.reduce_mean(
                loss_fn(test_labels[:len(val_preds)], val_preds)))

        avg_val = sum(val_losses.values()) / len(val_losses)

        history_train.append(avg_epoch_loss)
        history_val.append(avg_val)
        for s in RESOLUTION_SCALES:
            history_val_per_res[s].append(val_losses[s])

        current_step = (epoch + 1) * steps_per_epoch
        current_lr = float(schedule(current_step))
        epoch_time = int(time.time() - epoch_start)

        # Early stopping check
        improved = avg_val < (best_val_loss - min_delta)
        if improved:
            best_val_loss = avg_val
            best_weights = [w.numpy() for w in model.trainable_variables]
            epochs_without_improvement = 0
            marker = " *"  # mark best epoch
        else:
            epochs_without_improvement += 1
            marker = ""

        scale_str = " | ".join(
            f"{s:.2f}x:{val_losses[s]:.4f}" for s in RESOLUTION_SCALES)
        print(
            f'  [{phase_name}] Epoch {epoch+1}/{effective_max} '
            f'loss:{avg_epoch_loss:.4f} val:{avg_val:.4f} '
            f'lr:{current_lr:.2e} ({epoch_time}s) '
            f'[{scale_str}] '
            f'patience:{epochs_without_improvement}/{patience}{marker}')

        if epochs_without_improvement >= patience and not TEST_MODE:
            print(f"\n  Early stopping: no improvement for {patience} epochs.")
            print(f"  Best val loss: {best_val_loss:.4f}")
            break

    # Restore best weights
    if best_weights is not None and not TEST_MODE:
        for var, w in zip(model.trainable_variables, best_weights):
            var.assign(w)
        print(f"  Restored best weights (val loss {best_val_loss:.4f})")

    elapsed = int(time.time() - phase_start)
    final_epoch = len(history_train)
    print(f"  [{phase_name}] Completed: {final_epoch} epochs in {elapsed}s\n")

    return history_train, history_val, dict(history_val_per_res)


#========================================================
# Load datasets
#========================================================

print("\n========== Loading Phase 1 (all .db3 except Phase 2) ==========")
p1_lidar, p1_servo, p1_speed, p1_max = load_dataset(PHASE1_PATHS, down_sample_param)
print(f"Phase 1 total: {len(p1_lidar)} samples, max_speed={p1_max:.2f}")

print("\n========== Loading Phase 2 (selected files) ==========")
p2_lidar, p2_servo, p2_speed, p2_max = load_dataset(PHASE2_PATHS, down_sample_param)
print(f"Phase 2 total: {len(p2_lidar)} samples, max_speed={p2_max:.2f}")

global_max_speed = max(p1_max, p2_max)
global_min_speed = 0
print(f"\nGlobal speed range: [{global_min_speed}, {global_max_speed:.2f}]")

phase1_data = prepare_split(p1_lidar, p1_servo, p1_speed, global_max_speed)
phase2_data = prepare_split(p2_lidar, p2_servo, p2_speed, global_max_speed)

num_lidar_range_values = phase1_data['train_lidar'].shape[1]
assert phase2_data['train_lidar'].shape[1] == num_lidar_range_values
print(f'num_lidar_range_values: {num_lidar_range_values}')
for s in RESOLUTION_SCALES:
    print(f'  Scale {s:.2f}x -> {int(round(num_lidar_range_values * s))} input points')

#======================================================
# Build Model
#======================================================

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(24, 10, strides=4, activation='relu', input_shape=(None, 1)),
    ResolutionAwareBatchNormalization(RESOLUTION_SCALES, name='rabn_1'),

    tf.keras.layers.Conv1D(36, 8, strides=4, activation='relu'),
    ResolutionAwareBatchNormalization(RESOLUTION_SCALES, name='rabn_2'),

    tf.keras.layers.Conv1D(48, 4, strides=2, activation='relu'),
    ResolutionAwareBatchNormalization(RESOLUTION_SCALES, name='rabn_3'),

    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    ResolutionAwareBatchNormalization(RESOLUTION_SCALES, name='rabn_4'),

    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    ResolutionAwareBatchNormalization(RESOLUTION_SCALES, name='rabn_5'),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='tanh'),
])

model(np.zeros((1, num_lidar_range_values, 1), dtype=np.float32))
bn_native_lengths = patch_bn_native_lengths(model, num_lidar_range_values)
print("BN layer native lengths:", bn_native_lengths)
print(model.summary())

#======================================================
# Phase 1: Train on all .db3 EXCEPT Phase 2 files
#======================================================

p1_train, p1_val, p1_val_res = run_training_phase(
    model, phase1_data, MAX_EPOCHS, lr, batch_size, warmup_epochs,
    PATIENCE, MIN_DELTA,
    phase_name="Phase 1 (general)",
)

#======================================================
# Phase 2: Train on selected files — fresh schedule,
# same base LR, full warmup + decay cycle
#======================================================

p2_train, p2_val, p2_val_res = run_training_phase(
    model, phase2_data, MAX_EPOCHS, lr, batch_size, warmup_epochs,
    PATIENCE, MIN_DELTA,
    phase_name="Phase 2 (selected)",
)

#======================================================
# BN Calibration (on Phase 2 data — the fine-tuning set)
#======================================================

if CALIBRATE_BN:
    print("Running BN calibration sweep on Phase 2 data...")
    calibrate_batch_norm(
        model, phase2_data['train_lidar'], RESOLUTION_SCALES,
        batch_size, 1 if TEST_MODE else CALIBRATION_MAX_BATCHES)

#======================================================
# Loss Plot
#======================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.set_title(f'Phase 1 (general) — {len(p1_train)} epochs')
ax1.plot(p1_train, label='Train')
ax1.plot(p1_val, label='Val (avg)', linewidth=2)
for s in RESOLUTION_SCALES:
    ax1.plot(p1_val_res[s], label=f'Val {s:.2f}x', linestyle='--')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.set_title(f'Phase 2 (selected) — {len(p2_train)} epochs')
ax2.plot(p2_train, label='Train')
ax2.plot(p2_val, label='Val (avg)', linewidth=2)
for s in RESOLUTION_SCALES:
    ax2.plot(p2_val_res[s], label=f'Val {s:.2f}x', linestyle='--')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig(loss_figure_path)
plt.close()
print(f"Loss plot saved to {loss_figure_path}")

#======================================================
# Model Evaluation (Phase 2 test set)
#======================================================

print("\n==========================================")
print("Model Evaluation (Phase 2 test set)")
print("==========================================")

test_lidar_final = phase2_data['test_lidar']
test_labels_final = phase2_data['test_labels']

for scale_idx, scale in enumerate(RESOLUTION_SCALES):
    set_model_resolution(model, scale_idx)
    resized_test = resize_lidar_1d(test_lidar_final, scale)
    resized_input = np.expand_dims(resized_test, -1).astype(np.float32)

    preds = model(resized_input, training=False).numpy()
    hl = huber_loss(test_labels_final, preds)
    speed_hl = huber_loss(test_labels_final[:, 1], preds[:, 1])
    servo_hl = huber_loss(test_labels_final[:, 0], preds[:, 0])

    print(f'\nResolution {scale:.2f}x:')
    print(f'  Overall Huber Loss: {hl:.4f}')
    print(f'  Speed Huber Loss:   {speed_hl:.4f}')
    print(f'  Servo Huber Loss:   {servo_hl:.4f}')

#======================================================
# TFLite Export
#======================================================

print("\n==========================================")
print("TFLite Export")
print("==========================================")

@tf.function(input_signature=[
    tf.TensorSpec([1, None, 1], tf.float32, name='lidar'),
])
def serve(lidar_in):
    return model(lidar_in, training=False)

concrete_func = serve.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
tflite_path = ('./Benchmark/f1tenth_benchmarks/zarrar/'
               + model_name + '_noquantized.tflite')
os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"{model_name}_noquantized.tflite saved.")

rep_32 = phase2_data['train_lidar'][:200].astype(np.float32)
rep_32 = np.expand_dims(rep_32, -1)

def representative_data_gen():
    for i in range(len(rep_32)):
        yield [rep_32[i:i+1]]

converter_q = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
converter_q.representative_dataset = representative_data_gen
converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
quantized_tflite_model = converter_q.convert()

tflite_q_path = ('./Benchmark/f1tenth_benchmarks/zarrar/'
                 + model_name + '_int8.tflite')
with open(tflite_q_path, 'wb') as f:
    f.write(quantized_tflite_model)
print(f"{model_name}_int8.tflite saved.")

print('TFLite models saved.')

#======================================================
# Evaluate TFLite Model
#======================================================

def evaluate_model(model_path, test_lidar_data, test_labels):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    input_index = input_details[0]["index"]
    output_details = interpreter.get_output_details()

    first_shape = [1, len(test_lidar_data[0]), 1]
    interpreter.resize_tensor_input(input_index, first_shape)
    interpreter.allocate_tensors()
    current_len = first_shape[1]

    output_servo = []
    output_speed = []
    period = 1.0 / hz
    inference_times_micros = []

    for scan in test_lidar_data:
        scan_input = np.expand_dims(scan, axis=-1).astype(np.float32)
        scan_input = np.expand_dims(scan_input, axis=0)

        scan_len = scan_input.shape[1]
        if scan_len != current_len:
            interpreter.resize_tensor_input(input_index, list(scan_input.shape))
            interpreter.allocate_tensors()
            current_len = scan_len

        ts = time.time()
        interpreter.set_tensor(input_index, scan_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        dur = time.time() - ts

        inference_times_micros.append(dur * 1e6)
        if dur > period:
            print("%.3f: took %.2f us - deadline miss." % (dur, dur * 1e6))

        output_servo.append(output[0, 0])
        output_speed.append(output[0, 1])

    output_servo = np.asarray(output_servo)
    output_speed = np.asarray(output_speed)
    y_pred = np.concatenate(
        (output_servo[:, np.newaxis], output_speed[:, np.newaxis]), axis=1)

    arr = np.array(inference_times_micros)
    perc99 = np.percentile(arr, 99)
    arr_clipped = arr[arr < perc99]
    print(f"Model: {model_path}")
    print(f"Average Inference Time: {np.mean(arr_clipped):.2f} us")
    print(f"Maximum Inference Time: {np.max(arr_clipped):.2f} us")

    return y_pred, inference_times_micros


all_inference_times_micros = []
for m_name in model_files:
    if not os.path.exists(m_name):
        print(f"Skipping {m_name} (not found)")
        continue
    y_pred, inf_times = evaluate_model(m_name, test_lidar_final, test_labels_final)
    all_inference_times_micros.append(inf_times)
    print(f'Huber Loss for {m_name}: {huber_loss(test_labels_final, y_pred)}\n')

if all_inference_times_micros:
    plt.figure()
    for inf_times in all_inference_times_micros:
        arr = np.array(inf_times)
        perc99 = np.percentile(arr, 99)
        plt.plot(arr[arr < perc99])
    plt.xlabel('Inference Iteration')
    plt.ylabel('Inference Time (microseconds)')
    plt.title('Inference Time per Iteration')
    plt.legend(model_files[:len(all_inference_times_micros)])
    plt.savefig('./Figures/inference_times.png')
    plt.close()

print('End')