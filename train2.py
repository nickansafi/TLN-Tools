# Requirement Library
import os
import glob
from dataclasses import dataclass, field
from typing import List, Optional
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

# ========================================================
# Run configuration
#
# Each RunConfig is one independent training run that produces
# its own .tflite pair. Mix and match as needed.
#
#   name:              short label used in filenames and logs
#   resolution_scales: one or more scales to train at
#   shared_affine:     only meaningful when len(scales) > 1;
#                      must be None (or omitted) for single-scale
# ========================================================

@dataclass
class RunConfig:
    name: str
    resolution_scales: List[float]
    shared_affine: Optional[bool] = None  # None => single-scale, N/A

    def __post_init__(self):
        if len(self.resolution_scales) == 1:
            if self.shared_affine is not None:
                raise ValueError(
                    f"RunConfig {self.name!r}: shared_affine must be None "
                    f"for single-scale configs (got {self.shared_affine})"
                )
        else:
            if self.shared_affine is None:
                raise ValueError(
                    f"RunConfig {self.name!r}: shared_affine must be True "
                    f"or False for multi-scale configs"
                )

    @property
    def is_multires(self) -> bool:
        return len(self.resolution_scales) > 1

    @property
    def num_scales(self) -> int:
        return len(self.resolution_scales)

    @property
    def file_suffix(self) -> str:
        """Suffix used inside exported filenames (no leading underscore)."""
        if self.is_multires:
            tag = 'sharedaffine' if self.shared_affine else 'perbankaffine'
            return f'{self.name}_{tag}'
        return self.name


# Edit this list to control what gets trained.
RUN_CONFIGS: List[RunConfig] = [
    RunConfig(name='multires',   resolution_scales=[1.0, 0.75], shared_affine=True),
    RunConfig(name='multires',   resolution_scales=[1.0, 0.75], shared_affine=False),
    RunConfig(name='single1.00', resolution_scales=[1.0]),
    RunConfig(name='single0.75', resolution_scales=[0.75]),
]

# ========================================================
# Other configuration
# ========================================================
CALIBRATE_BN = True
CALIBRATION_MAX_BATCHES = 200
MATCH = False
INCREMENTAL = False

TEST_MODE = False

SKIP_PHASE1 = True
PHASE2_SIMPLE_EPOCHS = 200
PHASE2_SIMPLE_EARLY_STOP = True   # True => early stop in simple mode
PHASE2_SIMPLE_PATIENCE = 5
PHASE2_SIMPLE_MIN_DELTA = 1e-5

USE_MIXED_PRECISION = False
USE_XLA = True
SHUFFLE_BUFFER = 10000

DATASET_ROOT = './Dataset'

PATIENCE = 5
MIN_DELTA = 1e-5
MAX_EPOCHS = 200

PHASE2_FILENAMES = [
    'jfr1.db3',
    'jfr2.db3',
    'jfrv5_opp.db3',
    'jfrv6_opp.db3',
    'Forza_GLC_smile_PP_edgecases_0.db3',
]

ALL_DB3 = sorted(glob.glob(os.path.join(DATASET_ROOT, '**', '*.db3'), recursive=True))

PHASE2_PATHS = []
for db3_path in ALL_DB3:
    if os.path.basename(db3_path) in PHASE2_FILENAMES:
        PHASE2_PATHS.append(db3_path)

found_names = {os.path.basename(p) for p in PHASE2_PATHS}
missing = set(PHASE2_FILENAMES) - found_names
if missing:
    print(f"WARNING: Phase 2 files not found: {missing}")

phase2_abs = {os.path.abspath(p) for p in PHASE2_PATHS}
PHASE1_PATHS = [p for p in ALL_DB3 if os.path.abspath(p) not in phase2_abs]

print(f"Phase 1 datasets ({len(PHASE1_PATHS)}) — excludes Phase 2:")
for p in PHASE1_PATHS:
    print(f"  {p}")
print(f"Phase 2 datasets ({len(PHASE2_PATHS)}):")
for p in PHASE2_PATHS:
    print(f"  {p}")

down_sample_param = 1
lr = 5e-5
batch_size = 64
warmup_epochs = 2
hz = 40

model_name = 'TLN'
EXPORT_DIR = './Benchmark/f1tenth_benchmarks/zarrar/'
FIGURES_DIR = './Figures'


def tflite_paths_for(run: RunConfig):
    suffix = run.file_suffix
    return (
        os.path.join(EXPORT_DIR, f'{model_name}_{suffix}_noquantized.tflite'),
        os.path.join(EXPORT_DIR, f'{model_name}_{suffix}_int8.tflite'),
    )


def loss_figure_path_for(run: RunConfig) -> str:
    return os.path.join(FIGURES_DIR, f'loss_curve_{run.file_suffix}.png')


# ========================================================
# Mixed Precision Setup
# ========================================================

if USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision enabled: compute={policy.compute_dtype}, "
          f"variable={policy.variable_dtype}")

# ========================================================
# Helpers
# ========================================================

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
    loss = np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
    return np.mean(loss)


# ========================================================
# Learning Rate Schedule
# ========================================================

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
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


# ========================================================
# Resolution-Aware Batch Normalization
#
# Takes two inputs: [x, scale_index].
#   x:            (N, L, C) feature map
#   scale_index:  (N,) int32 — only scale_index[0] is read.
#
# When num_scales == 1, shared_affine is irrelevant (the two
# storage layouts collapse to the same thing), and we just use
# the simpler shared layout.
# ========================================================

class ResolutionAwareBatchNormalization(tf.keras.layers.Layer):

    def __init__(self, num_scales, momentum=0.1, epsilon=1e-5,
                 shared_affine=True, **kwargs):
        super().__init__(**kwargs)
        self.num_scales = int(num_scales)
        self.momentum = momentum
        self.epsilon = epsilon
        # For single-scale runs the distinction is vacuous — force to True
        # so the weight layout is unambiguous.
        self.shared_affine = True if self.num_scales == 1 else bool(shared_affine)

    def build(self, input_shape):
        x_shape = input_shape[0]
        nf = x_shape[-1]

        if self.shared_affine:
            self.gamma = self.add_weight(
                name='gamma', shape=(nf,), initializer='ones', trainable=True)
            self.beta = self.add_weight(
                name='beta', shape=(nf,), initializer='zeros', trainable=True)
        else:
            self.all_gammas = self.add_weight(
                name='all_gammas', shape=(self.num_scales, nf),
                initializer='ones', trainable=True)
            self.all_betas = self.add_weight(
                name='all_betas', shape=(self.num_scales, nf),
                initializer='zeros', trainable=True)

        self.all_running_means = self.add_weight(
            name='all_running_means', shape=(self.num_scales, nf),
            initializer='zeros', trainable=False)
        self.all_running_vars = self.add_weight(
            name='all_running_vars', shape=(self.num_scales, nf),
            initializer='ones', trainable=False)
        self.all_num_batches = self.add_weight(
            name='all_num_batches', shape=(self.num_scales,),
            initializer='zeros', dtype=tf.float32, trainable=False)
        super().build(input_shape)

    def _get_affine(self, idx):
        if self.shared_affine:
            return self.gamma, self.beta
        return tf.gather(self.all_gammas, idx), tf.gather(self.all_betas, idx)

    def call(self, inputs, training=False):
        x, scale_index = inputs
        idx = scale_index[0]

        if training:
            mean = tf.reduce_mean(x, axis=[0, 1])
            var = tf.math.reduce_variance(x, axis=[0, 1])

            m = self.momentum
            if m is None:
                count = tf.gather(self.all_num_batches, idx)
                m = 1.0 / (tf.cast(count, tf.float32) + 1.0)

            old_mean = tf.gather(self.all_running_means, idx)
            old_var = tf.gather(self.all_running_vars, idx)
            new_mean = old_mean * (1.0 - m) + mean * m
            new_var = old_var * (1.0 - m) + var * m

            self.all_running_means.assign(
                tf.tensor_scatter_nd_update(
                    self.all_running_means, [[idx]],
                    tf.expand_dims(new_mean, 0)))
            self.all_running_vars.assign(
                tf.tensor_scatter_nd_update(
                    self.all_running_vars, [[idx]],
                    tf.expand_dims(new_var, 0)))
            self.all_num_batches.assign(
                tf.tensor_scatter_nd_update(
                    self.all_num_batches, [[idx]],
                    [tf.gather(self.all_num_batches, idx) + 1.0]))
        else:
            mean = tf.gather(self.all_running_means, idx)
            var = tf.gather(self.all_running_vars, idx)

        gamma, beta = self._get_affine(idx)
        x_norm = (x - mean) / tf.sqrt(var + self.epsilon)
        return gamma * x_norm + beta

    def reset_running_stats(self):
        self.all_running_means.assign(tf.zeros_like(self.all_running_means))
        self.all_running_vars.assign(tf.ones_like(self.all_running_vars))
        self.all_num_batches.assign(tf.zeros_like(self.all_num_batches))

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_scales': self.num_scales,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'shared_affine': self.shared_affine,
        })
        return config

# ========================================================
# Multi-Resolution Training Wrapper
#
# Works for any number of scales >= 1. Single-scale runs just
# execute one forward/backward per step, which is equivalent
# to ordinary training.
# ========================================================

class MultiResolutionTrainer(tf.keras.Model):
    def __init__(self, inner_model, resolution_scales,
                 use_mixed_precision=False, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner_model
        self.resolution_scales = resolution_scales
        self.num_scales_f = float(len(resolution_scales))
        self._use_mp = use_mixed_precision
        self.loss_fn = tf.keras.losses.Huber()

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.scale_loss_trackers = [
            tf.keras.metrics.Mean(name=f'loss_{s:.2f}x')
            for s in resolution_scales
        ]

    @property
    def metrics(self):
        return [self.loss_tracker] + self.scale_loss_trackers

    def call(self, inputs, training=False):
        return self.inner(inputs, training=training)

    def _scale_idx_tensor(self, x_scale, scale_idx):
        bs = tf.shape(x_scale)[0]
        return tf.fill([bs], tf.constant(scale_idx, dtype=tf.int32))

    def train_step(self, data):
        scale_inputs, labels = data

        with tf.GradientTape() as tape:
            total_loss = tf.constant(0.0)
            for scale_idx in range(len(self.resolution_scales)):
                x_scale = scale_inputs[scale_idx]
                idx_tensor = self._scale_idx_tensor(x_scale, scale_idx)
                preds = self.inner([x_scale, idx_tensor], training=True)
                total_loss = total_loss + tf.reduce_mean(
                    self.loss_fn(labels, preds))
            avg_loss = total_loss / self.num_scales_f

            if self._use_mp:
                scaled_loss = self.optimizer.get_scaled_loss(avg_loss)

        if self._use_mp:
            grads = tape.gradient(scaled_loss, self.inner.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(grads)
        else:
            grads = tape.gradient(avg_loss, self.inner.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads, self.inner.trainable_variables))
        self.loss_tracker.update_state(avg_loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data):
        scale_inputs, labels = data

        total_loss = tf.constant(0.0)
        for scale_idx in range(len(self.resolution_scales)):
            x_scale = scale_inputs[scale_idx]
            idx_tensor = self._scale_idx_tensor(x_scale, scale_idx)
            preds = self.inner([x_scale, idx_tensor], training=False)
            scale_loss = tf.reduce_mean(self.loss_fn(labels, preds))
            total_loss = total_loss + scale_loss
            self.scale_loss_trackers[scale_idx].update_state(scale_loss)

        avg_loss = total_loss / self.num_scales_f
        self.loss_tracker.update_state(avg_loss)

        results = {'loss': self.loss_tracker.result()}
        for i, s in enumerate(self.resolution_scales):
            results[f'loss_{s:.2f}x'] = self.scale_loss_trackers[i].result()
        return results


# ========================================================
# BN Calibration
# ========================================================

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
            resized = resize_lidar_1d(batch, scale)
            resized = np.expand_dims(resized, axis=-1).astype(np.float32)
            idx_arr = np.full([len(resized)], scale_idx, dtype=np.int32)
            model([resized, idx_arr], training=True)
        batches_processed += 1
        if max_batches is not None and batches_processed >= max_batches:
            break

    for layer, m in zip(bn_layers, original_momenta):
        layer.momentum = m
    print(f"BN calibration complete: {batches_processed} batches, "
          f"{len(resolution_scales)} scales")


# ========================================================
# Dataset Loading
# ========================================================

def load_dataset(db3_paths, down_sample_param=1):
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


# ========================================================
# Training Phase
# ========================================================

def run_training_phase(model, data, max_epochs, base_lr, batch_size,
                       warmup_epochs, patience, min_delta,
                       run: RunConfig, phase_name="Phase"):

    train_lidar = data['train_lidar']
    train_labels = data['train_labels']
    test_lidar = data['test_lidar']
    test_labels = data['test_labels']

    effective_max = 1 if TEST_MODE else max_epochs

    print(f"  Precomputing resized data for {run.num_scales} scale(s)...")
    train_per_scale = []
    test_per_scale = []
    for scale in run.resolution_scales:
        tr = resize_lidar_1d(train_lidar, scale)
        train_per_scale.append(np.expand_dims(tr, -1).astype(np.float32))
        te = resize_lidar_1d(test_lidar, scale)
        test_per_scale.append(np.expand_dims(te, -1).astype(np.float32))

    train_labels_f32 = train_labels.astype(np.float32)
    test_labels_f32 = test_labels.astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tuple(train_per_scale), train_labels_f32))
    train_ds = (train_ds
                .shuffle(min(len(train_lidar), SHUFFLE_BUFFER))
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = tf.data.Dataset.from_tensor_slices(
        (tuple(test_per_scale), test_labels_f32))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = max(1, len(train_lidar) // batch_size)
    schedule = build_lr_schedule(
        base_lr, effective_max, steps_per_epoch, warmup_epochs)
    optimizer = Adam(learning_rate=schedule)
    if USE_MIXED_PRECISION:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    trainer = MultiResolutionTrainer(
        model, run.resolution_scales, USE_MIXED_PRECISION)
    trainer.compile(optimizer=optimizer, jit_compile=USE_XLA)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"\n{'=' * 60}")
    print(f" {phase_name}: up to {effective_max} epochs, base_lr={base_lr:.2e}")
    print(f" Early stopping: patience={patience}, min_delta={min_delta:.1e}")
    if TEST_MODE:
        print(f" *** TEST MODE: 1 epoch only ***")
    print(f" {len(train_lidar)} train / {len(test_lidar)} test samples")
    print(f" Warmup: {warmup_epochs} epochs "
          f"({warmup_epochs * steps_per_epoch} steps)")
    print(f" Steps/epoch: {steps_per_epoch}")
    print(f" XLA={USE_XLA} | mixed_prec={USE_MIXED_PRECISION}")
    print(f" scales={run.resolution_scales}")
    if run.is_multires:
        print(f" shared_affine={run.shared_affine}")
    else:
        print(f" shared_affine=N/A (single-scale)")
    print(f"{'=' * 60}\n")

    history = trainer.fit(
        train_ds,
        validation_data=val_ds,
        epochs=effective_max,
        callbacks=callbacks,
        verbose=1,
    )

    h = history.history
    train_losses = h.get('loss', [])
    val_losses = h.get('val_loss', [])
    val_per_res = {}
    for s in run.resolution_scales:
        val_per_res[s] = h.get(f'val_loss_{s:.2f}x', [])

    final_epoch = len(train_losses)
    print(f"  [{phase_name}] Completed: {final_epoch} epochs\n")

    return train_losses, val_losses, val_per_res


# ========================================================
# Load datasets (once, shared across all runs)
# ========================================================

if not SKIP_PHASE1:
    print("\n========== Loading Phase 1 (all .db3 except Phase 2) ==========")
    p1_lidar, p1_servo, p1_speed, p1_max = load_dataset(
        PHASE1_PATHS, down_sample_param)
    print(f"Phase 1 total: {len(p1_lidar)} samples, max_speed={p1_max:.2f}")
else:
    print("\n========== SKIP_PHASE1=True — skipping Phase 1 data load ==========")
    p1_max = 0.0

print("\n========== Loading Phase 2 (selected files) ==========")
p2_lidar, p2_servo, p2_speed, p2_max = load_dataset(
    PHASE2_PATHS, down_sample_param)
print(f"Phase 2 total: {len(p2_lidar)} samples, max_speed={p2_max:.2f}")

global_max_speed = max(p1_max, p2_max)
global_min_speed = 0
print(f"\nGlobal speed range: [{global_min_speed}, {global_max_speed:.2f}]")

if not SKIP_PHASE1:
    phase1_data = prepare_split(p1_lidar, p1_servo, p1_speed, global_max_speed)
else:
    phase1_data = None

phase2_data = prepare_split(p2_lidar, p2_servo, p2_speed, global_max_speed)

num_lidar_range_values = phase2_data['train_lidar'].shape[1]
if not SKIP_PHASE1:
    assert phase1_data['train_lidar'].shape[1] == num_lidar_range_values
print(f'num_lidar_range_values: {num_lidar_range_values}')

# ======================================================
# Build / Train / Export — one per RunConfig
# ======================================================

output_dtype = 'float32' if USE_MIXED_PRECISION else None


def build_model(run: RunConfig):
    lidar_in = tf.keras.Input(shape=(None, 1), name='lidar')
    scale_idx_in = tf.keras.Input(shape=(), dtype=tf.int32, name='scale_index')

    def bn(name):
        return ResolutionAwareBatchNormalization(
            run.num_scales,
            shared_affine=(run.shared_affine if run.is_multires else True),
            name=name,
        )

    x = tf.keras.layers.Conv1D(24, 10, strides=4, activation='relu')(lidar_in)
    x = bn('rabn_1')([x, scale_idx_in])

    x = tf.keras.layers.Conv1D(36, 8, strides=4, activation='relu')(x)
    x = bn('rabn_2')([x, scale_idx_in])

    x = tf.keras.layers.Conv1D(48, 4, strides=2, activation='relu')(x)
    x = bn('rabn_3')([x, scale_idx_in])

    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    x = bn('rabn_4')([x, scale_idx_in])

    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    x = bn('rabn_5')([x, scale_idx_in])

    x = GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    out = tf.keras.layers.Dense(2, activation='tanh', dtype=output_dtype)(x)

    return tf.keras.Model(
        [lidar_in, scale_idx_in], out, name=f'TLN_{run.file_suffix}')


def plot_losses(p1_train, p1_val, p1_val_res,
                p2_train, p2_val, p2_val_res,
                run: RunConfig):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    suffix = run.file_suffix

    if p1_train:
        ax1.set_title(f'Phase 1 ({suffix}) — {len(p1_train)} epochs')
        ax1.plot(p1_train, label='Train')
        ax1.plot(p1_val, label='Val (avg)', linewidth=2)
        for s in run.resolution_scales:
            ax1.plot(p1_val_res[s], label=f'Val {s:.2f}x', linestyle='--')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
    else:
        ax1.set_title('Phase 1 — skipped')
        ax1.text(0.5, 0.5, 'Skipped', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=14, color='gray')

    ax2.set_title(f'Phase 2 ({suffix}) — {len(p2_train)} epochs')
    ax2.plot(p2_train, label='Train')
    ax2.plot(p2_val, label='Val (avg)', linewidth=2)
    for s in run.resolution_scales:
        ax2.plot(p2_val_res[s], label=f'Val {s:.2f}x', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = loss_figure_path_for(run)
    plt.savefig(out_path)
    plt.close()
    print(f"Loss plot saved to {out_path}")


def evaluate_tflite(model_path, test_lidar_data, test_labels,
                    scale, scale_index):
    interpreter = tf.lite.Interpreter(model_path=model_path)

    lidar_input_index = scale_input_index = None
    for d in interpreter.get_input_details():
        if d['dtype'] == np.int32:
            scale_input_index = d['index']
        else:
            lidar_input_index = d['index']
    if lidar_input_index is None or scale_input_index is None:
        raise RuntimeError(
            "TFLite model must expose one float lidar input "
            "and one int32 scale_index input")

    output_details = interpreter.get_output_details()

    # Resize test lidar to match the scale we're evaluating at.
    resized_all = resize_lidar_1d(test_lidar_data, scale)
    first_shape = [1, resized_all.shape[1], 1]
    interpreter.resize_tensor_input(lidar_input_index, first_shape)
    interpreter.allocate_tensors()
    current_len = first_shape[1]

    idx_tensor = np.array([scale_index], dtype=np.int32)

    output_servo = []
    output_speed = []
    period = 1.0 / hz
    inference_times_micros = []

    for scan in resized_all:
        scan_input = np.expand_dims(scan, axis=-1).astype(np.float32)
        scan_input = np.expand_dims(scan_input, axis=0)

        scan_len = scan_input.shape[1]
        if scan_len != current_len:
            interpreter.resize_tensor_input(lidar_input_index, list(scan_input.shape))
            interpreter.allocate_tensors()
            current_len = scan_len

        ts = time.time()
        interpreter.set_tensor(lidar_input_index, scan_input)
        interpreter.set_tensor(scale_input_index, idx_tensor)
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
    print(f"Model: {model_path} @ scale={scale}")
    print(f"Average Inference Time: {np.mean(arr_clipped):.2f} us")
    print(f"Maximum Inference Time: {np.max(arr_clipped):.2f} us")

    return y_pred, inference_times_micros


def run_one(run: RunConfig):
    suffix = run.file_suffix
    print("\n\n" + "#" * 70)
    print(f"# Run: {suffix}")
    print(f"#   scales={run.resolution_scales}")
    if run.is_multires:
        print(f"#   shared_affine={run.shared_affine}")
    else:
        print(f"#   shared_affine=N/A")
    print("#" * 70)

    model = build_model(run)

    _dummy_lidar = np.zeros((1, num_lidar_range_values, 1), dtype=np.float32)
    _dummy_idx = np.zeros((1,), dtype=np.int32)
    model([_dummy_lidar, _dummy_idx])
    print(model.summary())

    # ----- Phase 1 -----
    if SKIP_PHASE1:
        print("\n*** SKIP_PHASE1=True — skipping Phase 1 training ***")
        p1_train, p1_val, p1_val_res = [], [], {}
    else:
        p1_train, p1_val, p1_val_res = run_training_phase(
            model, phase1_data, MAX_EPOCHS, lr, batch_size, warmup_epochs,
            PATIENCE, MIN_DELTA, run,
            phase_name=f"Phase 1 ({suffix})",
        )

    # ----- Phase 2 -----
    if SKIP_PHASE1:
        if PHASE2_SIMPLE_EARLY_STOP:
            simple_patience = PHASE2_SIMPLE_PATIENCE
            simple_min_delta = PHASE2_SIMPLE_MIN_DELTA
        else:
            simple_patience = PHASE2_SIMPLE_EPOCHS
            simple_min_delta = 0

        p2_train, p2_val, p2_val_res = run_training_phase(
            model, phase2_data, PHASE2_SIMPLE_EPOCHS, lr, batch_size,
            warmup_epochs=0, patience=simple_patience, min_delta=simple_min_delta,
            run=run, phase_name=f"Phase 2 simple ({suffix})",
        )

    # ----- BN Calibration -----
    if CALIBRATE_BN:
        print(f"Running BN calibration sweep on Phase 2 data ({suffix})...")
        calibrate_batch_norm(
            model, phase2_data['train_lidar'], run.resolution_scales,
            batch_size, 1 if TEST_MODE else CALIBRATION_MAX_BATCHES)

    # ----- Loss plot -----
    plot_losses(p1_train, p1_val, p1_val_res,
                p2_train, p2_val, p2_val_res, run)

    # ----- Keras-level eval -----
    print("\n==========================================")
    print(f"Model Evaluation (Phase 2 test set) — {suffix}")
    print("==========================================")

    test_lidar_final = phase2_data['test_lidar']
    test_labels_final = phase2_data['test_labels']

    for scale_idx, scale in enumerate(run.resolution_scales):
        resized_test = resize_lidar_1d(test_lidar_final, scale)
        resized_input = np.expand_dims(resized_test, -1).astype(np.float32)
        idx_arr = np.full([len(resized_input)], scale_idx, dtype=np.int32)

        preds = model([resized_input, idx_arr], training=False).numpy()
        hl = huber_loss(test_labels_final, preds)
        speed_hl = huber_loss(test_labels_final[:, 1], preds[:, 1])
        servo_hl = huber_loss(test_labels_final[:, 0], preds[:, 0])

        print(f'\nResolution {scale:.2f}x (scale_index={scale_idx}):')
        print(f'  Overall Huber Loss: {hl:.4f}')
        print(f'  Speed Huber Loss:   {speed_hl:.4f}')
        print(f'  Servo Huber Loss:   {servo_hl:.4f}')

    # ----- TFLite export -----
    print("\n==========================================")
    print(f"TFLite Export — {suffix}")
    print("==========================================")

    if USE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('float32')

    @tf.function(input_signature=[
        tf.TensorSpec([1, None, 1], tf.float32, name='lidar'),
        tf.TensorSpec([1], tf.int32, name='scale_index'),
    ])
    def serve(lidar_in, scale_index):
        return model([lidar_in, scale_index], training=False)

    concrete_func = serve.get_concrete_function()

    tflite_path, tflite_q_path = tflite_paths_for(run)
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved: {tflite_path}")

    rep_base = phase2_data['train_lidar'][:200]

    def representative_data_gen():
        for scale_idx, scale in enumerate(run.resolution_scales):
            resized = resize_lidar_1d(rep_base, scale)
            resized = np.expand_dims(resized, -1).astype(np.float32)
            idx_val = np.array([scale_idx], dtype=np.int32)
            for i in range(len(resized)):
                yield [resized[i:i + 1], idx_val]

    converter_q = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_q.representative_dataset = representative_data_gen
    converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    quantized_tflite_model = converter_q.convert()

    with open(tflite_q_path, 'wb') as f:
        f.write(quantized_tflite_model)
    print(f"Saved: {tflite_q_path}")

    # ----- TFLite post-hoc eval at every trained scale -----
    print("\n==========================================")
    print(f"TFLite Evaluation — {suffix}")
    print("==========================================")

    all_inference_times_micros = []
    for m_path in (tflite_path, tflite_q_path):
        if not os.path.exists(m_path):
            print(f"Skipping {m_path} (not found)")
            continue
        for scale_idx, scale in enumerate(run.resolution_scales):
            y_pred, inf_times = evaluate_tflite(
                m_path, test_lidar_final, test_labels_final,
                scale=scale, scale_index=scale_idx)
            all_inference_times_micros.append(
                (f"{os.path.basename(m_path)} @ {scale:.2f}x", inf_times))
            print(f'Huber Loss for {m_path} @ {scale:.2f}x: '
                  f'{huber_loss(test_labels_final, y_pred)}\n')

    if all_inference_times_micros:
        plt.figure()
        labels_for_legend = []
        for label, inf_times in all_inference_times_micros:
            arr = np.array(inf_times)
            perc99 = np.percentile(arr, 99)
            plt.plot(arr[arr < perc99])
            labels_for_legend.append(label)
        plt.xlabel('Inference Iteration')
        plt.ylabel('Inference Time (microseconds)')
        plt.title(f'Inference Time per Iteration ({suffix})')
        plt.legend(labels_for_legend)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, f'inference_times_{suffix}.png')
        plt.savefig(fig_path)
        plt.close()
        print(f"Inference-time plot saved to {fig_path}")


for run_cfg in RUN_CONFIGS:
    run_one(run_cfg)

print('\nEnd')
