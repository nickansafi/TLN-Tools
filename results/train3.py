#!/usr/bin/env python3
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ROS 2 bag imports (same as your scripts)
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

# =========================
# Seed & determinism
# =========================
np.random.seed(1)
tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# =========================
# Config
# =========================
DATASET_PATHS = [
    "./Dataset/out.db3",
    "./Dataset/f2.db3",
    "./Dataset/f4.db3",
]
SKIPS = [1, 2, 4]
BATCH_SIZE = 64
LR = 5e-5
EPOCHS = 60
MODEL_NAME = "TLN_condbn"
LOSS_FIG_PATH = f"./Figures/{MODEL_NAME}_loss.png"
MODEL_SAVE_PATH = f"./Benchmark/f1tenth_benchmarks/zarrar/{MODEL_NAME}.h5"
CLIP_RANGE = 10.0
os.makedirs("./Models", exist_ok=True)
os.makedirs("./Figures", exist_ok=True)

# =========================
# make it like train.py
# =========================
max_speed = 0.0
min_speed = 0.0   # stays 0, just like the original train.py :contentReference[oaicite:2]{index=2}

# =========================
# Utils
# =========================
def linear_map(x, x_min, x_max, y_min, y_max):
    # match train.py style: no +1e-8, plain formula :contentReference[oaicite:3]{index=3}
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

def read_ros2_bag(bag_path, skip):
    """
    Read one ros2 bag with a given skip.
    Now also updates the global max_speed exactly like train.py
    (min_speed stays 0). :contentReference[oaicite:4]{index=4}
    """
    global max_speed, min_speed

    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts    = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_data, servo_data, speed_data = [], [], []

    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()

        if topic == 'Lidar':
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(msg.ranges[::skip], posinf=0.0, neginf=0.0)
            cleaned[cleaned > CLIP_RANGE] = CLIP_RANGE
            cleaned = cleaned / CLIP_RANGE
            lidar_data.append(cleaned.astype(np.float32))

        elif topic == 'Ackermann':
            msg = deserialize_message(serialized_msg, AckermannDriveStamped)
            sv = float(msg.drive.steering_angle)
            sp = float(msg.drive.speed)

            # match train.py: start at 0, only update max_speed with an if
            if sp > max_speed:
                max_speed = sp

            servo_data.append(sv)
            speed_data.append(sp)

    return (
        np.array(lidar_data, dtype=object),
        np.array(servo_data, dtype=np.float32),
        np.array(speed_data, dtype=np.float32),
    )

# =========================
# Conditional BatchNorm (unchanged)
# =========================
class ConditionalBatchNorm(tf.keras.layers.Layer):
    def __init__(self, num_res=3, **kwargs):
        super().__init__(**kwargs)
        self.num_res = num_res
        self.bn_low = tf.keras.layers.BatchNormalization()
        self.bn_mid = tf.keras.layers.BatchNormalization()
        self.bn_high = tf.keras.layers.BatchNormalization()

    def call(self, x, res_id, training=None):
        # res_id expected shape: (batch, 1) or (batch,)
        res_id_scalar = tf.squeeze(res_id)
        res_id_scalar = tf.cast(res_id_scalar, tf.int32)

        # We assume every batch has one resolution
        res_id_scalar = res_id[0]  # use first element if batch has one resolution

        def use_low():
            # tf.print("→ Using LOW resolution BN")
            return self.bn_low(x, training=training)

        def use_mid():
            # tf.print("→ Using MID resolution BN")
            return self.bn_mid(x, training=training)

        def use_high():
            # tf.print("→ Using HIGH resolution BN")
            return self.bn_high(x, training=training)

        # Nest tf.cond for 3 cases (avoids one-hots entirely)
        return tf.cond(
            res_id_scalar < 1,
            lambda: use_low(),
            lambda: tf.cond(
                res_id_scalar < 2,
                lambda: use_mid(),
                lambda: use_high()
            )
        )

def build_model(num_resolutions):
    """
    Functional-style version of the current TLN_condbn model.
    Same structure, but instead of only using GlobalAveragePooling1D,
    it concatenates both GlobalAveragePooling1D and GlobalMaxPooling1D outputs
    before feeding into the dense layers.
    """

    # Inputs
    lidar_in = tf.keras.Input(shape=(None, 1), name="lidar_input")
    res_in   = tf.keras.Input(shape=(1,), dtype=tf.int32, name="res_id")

    # --- Convolutional blocks (same as original) ---
    x = tf.keras.layers.Conv1D(64, 10, strides=4, activation='relu')(lidar_in)
    x = ConditionalBatchNorm(num_res=num_resolutions)(x, res_in)

    x = tf.keras.layers.Conv1D(128, 8, strides=4, activation='relu')(x)
    x = ConditionalBatchNorm(num_res=num_resolutions)(x, res_in)

    x = tf.keras.layers.Conv1D(256, 4, strides=2, activation='relu')(x)
    x = ConditionalBatchNorm(num_res=num_resolutions)(x, res_in)

    x = tf.keras.layers.Conv1D(256, 3, activation='relu')(x)
    x = ConditionalBatchNorm(num_res=num_resolutions)(x, res_in)

    # --- Combine average and max pooling ---
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    concat_pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    # --- Fully connected head (same structure) ---
    fc1 = tf.keras.layers.Dense(256, activation='relu')(concat_pooled)
    fc2 = tf.keras.layers.Dense(128, activation='relu')(fc1)
    fc3 = tf.keras.layers.Dense(32, activation='relu')(fc2)
    outputs = tf.keras.layers.Dense(2, activation='tanh')(fc3)

    # --- Build model ---
    model = tf.keras.Model(inputs=[lidar_in, res_in], outputs=outputs, name="TLN_condbn_func")
    return model

# =========================
# Load data from all bags & all skips
# =========================
groups = {}
all_lidars, all_servos, all_speeds = [], [], []

for pth in DATASET_PATHS:
    if not os.path.exists(pth):
        print(f"{pth} doesn't exist, skipping.")
        continue
    for skip in SKIPS:
        try:
            lidar_arr, servo_arr, speed_arr = read_ros2_bag(pth, skip)
        except Exception as e:
            continue

        for scan, sv, sp in zip(lidar_arr, servo_arr, speed_arr):
            L = len(scan)
            if L == 0:
                continue
            if L not in groups:
                groups[L] = {'lidar': [], 'servo': [], 'speed': []}
            groups[L]['lidar'].append(scan)
            groups[L]['servo'].append(sv)
            groups[L]['speed'].append(sp)

            # Keep global lists for normalization later
            all_lidars.append(scan)
            all_servos.append(sv)
            all_speeds.append(sp)

print(f"Found {len(groups)} distinct resolutions: {list(groups.keys())}")
print(f"Min_speed: {min_speed}")
print(f"Max_speed: {max_speed}")

sorted_lengths = sorted(groups.keys())
length_to_resid = {L: i for i, L in enumerate(sorted_lengths)}

# =========================
# Normalize speeds globally (like train.py)
# =========================
all_speeds = np.array(all_speeds, dtype=np.float32)
norm_all_speeds = linear_map(all_speeds, min_speed, max_speed, 0.0, 1.0)

# Map normalized speeds back into each group
start = 0
for L, data in groups.items():
    n = len(data['speed'])
    data['speed'] = norm_all_speeds[start:start+n]
    start += n

# =========================
# Build tf.data datasets
# =========================
datasets = []
for L, data in groups.items():
    lidar_np = np.array(data['lidar'], dtype=object)
    servo_np = np.array(data['servo'], dtype=np.float32)
    speed_np = np.array(data['speed'], dtype=np.float32)

    y = np.stack([servo_np, speed_np], axis=-1).astype(np.float32)

    n = len(lidar_np)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.85 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    res_id_val = length_to_resid[L]

    def make_ds(idxs):
        X_lidar = [np.expand_dims(lidar_np[i], -1).astype(np.float32) for i in idxs]
        X_resid = [np.array([res_id_val], dtype=np.int32) for _ in idxs]
        Y       = y[idxs]
        ds = tf.data.Dataset.from_tensor_slices(((X_lidar, X_resid), Y))
        ds = ds.shuffle(512).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(train_idx)
    val_ds   = make_ds(val_idx)
    datasets.append((train_ds, val_ds, L, res_id_val))

num_resolutions = len(sorted_lengths)

# =========================
# Build & compile model
# =========================
model = build_model(num_resolutions=num_resolutions)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
loss_fn = tf.keras.losses.Huber()
model.compile(optimizer=optimizer, loss=loss_fn)
model.summary()

# =========================
# Training loop
# =========================
history_losses = []
history_val_losses = []

start_time = time.time()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    order = np.arange(len(datasets))
    np.random.shuffle(order)
    for k in order:
        train_ds, val_ds, L, resid = datasets[k]
        print(f"  ▶ Resolution len={L} (id={resid})")
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            verbose=1,
        )
        history_losses.append(hist.history['loss'][-1])
        history_val_losses.append(hist.history.get('val_loss', [np.nan])[-1])

print(f"\nTraining time: {int(time.time() - start_time)} seconds")

# =========================
# Save model
# =========================
model.save(MODEL_SAVE_PATH)
print(f"Saved model to {MODEL_SAVE_PATH}")

# =========================
# Plot loss
# =========================
plt.figure()
plt.plot(history_losses, label="loss")
plt.plot(history_val_losses, label="val_loss")
plt.legend()
plt.xlabel("per-resolution step")
plt.ylabel("loss")
plt.title("Training loss trace")
plt.savefig(LOSS_FIG_PATH)
plt.close()
print(f"Saved loss figure to {LOSS_FIG_PATH}")
