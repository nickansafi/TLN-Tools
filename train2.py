#Requirement Library
import os
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

#========================================================
# Helpers
#========================================================

def compute_conv_output_length(length, kernel_size, stride):
    return (length - kernel_size) // stride + 1


def get_bn_native_lengths(model, input_length):
    """Compute the spatial dim each BN layer sees by walking
    the model's Conv1D and BN layers in order."""
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
    """Extract conv configs from the model and patch each BN layer's
    native_length and expected_lengths in place."""
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
# Global Data
#========================================================

lidar = []
servo = []
speed = []
test_lidar = []
test_servo = []
test_speed = []
model_name = 'TLN'
model_files = [
    './Benchmark/f1tenth_benchmarks/zarrar/' + model_name + '_noquantized.tflite',
    './Benchmark/f1tenth_benchmarks/zarrar/' + model_name + '_int8.tflite'
]
dataset_path = [
    './Dataset/out/out.db3',
    './Dataset/f2/f2.db3',
    './Dataset/f4/f4.db3'
]
loss_figure_path = './Figures/loss_curve.png'
down_sample_param = 1
lr = 5e-5
loss_function = 'huber'
batch_size = 64
num_epochs = 20
hz = 40

max_speed = 0
min_speed = 0

#========================================================
# Get Dataset
#========================================================

for pth in dataset_path:
    if not os.path.exists(pth):
        print(f"out.bag doesn't exist in {pth}")
        exit(0)

    lidar_data = []
    servo_data = []
    speed_data = []

    connection = sqlite3.connect(pth)
    cursor = connection.cursor()
    cursor.execute(
        "SELECT topics.name, topics.type, messages.data "
        "FROM messages JOIN topics ON messages.topic_id = topics.id"
    )
    for topic, type, rawdata in cursor:
        try:
            msg = typestore.deserialize_cdr(rawdata, type)
        except:
            continue
        if topic in ['Lidar', 'scan']:
            if len(msg.ranges) != 1081:
                continue
            ranges = msg.ranges[::down_sample_param]
            lidar_data.append(ranges)
        if topic in ['Ackermann', 'drive']:
            data = msg.drive.steering_angle
            s_data = msg.drive.speed
            servo_data.append(data)
            if s_data > max_speed:
                max_speed = s_data
            speed_data.append(s_data)
    connection.close()
    if len(set([len(lidar_data), len(servo_data), len(speed_data)])) != 1:
        continue

    lidar_data = np.array(lidar_data)
    servo_data = np.array(servo_data)
    speed_data = np.array(speed_data)

    shuffled_data = shuffle(
        np.concatenate((servo_data[:, np.newaxis], speed_data[:, np.newaxis]), axis=1),
        random_state=62,
    )
    shuffled_lidar_data = shuffle(lidar_data, random_state=62)

    train_ratio = 0.85
    train_samples = int(train_ratio * len(shuffled_lidar_data))
    x_train_bag = shuffled_lidar_data[:train_samples]
    x_test_bag = shuffled_lidar_data[train_samples:]

    y_train_bag = shuffled_data[:train_samples]
    y_test_bag = shuffled_data[train_samples:]

    lidar.extend(x_train_bag)
    servo.extend(y_train_bag[:, 0])
    speed.extend(y_train_bag[:, 1])

    test_lidar.extend(x_test_bag)
    test_servo.extend(y_test_bag[:, 0])
    test_speed.extend(y_test_bag[:, 1])

    print(f'\nData in {pth}:')
    print(f'Shape of Train Data --- Lidar: {len(lidar)}, Servo: {len(servo)}, '
          f'Speed: {len(speed)}')
    print(f'Shape of Test Data --- Lidar: {len(test_lidar)}, '
          f'Servo: {len(test_servo)}, Speed: {len(test_speed)}')

total_number_samples = len(lidar)
print(f'Overall Samples = {total_number_samples}')

lidar = np.asarray(lidar)
servo = np.asarray(servo)
speed = np.asarray(speed)
speed = linear_map(speed, min_speed, max_speed, 0, 1)
test_lidar = np.asarray(test_lidar)
test_servo = np.asarray(test_servo)
test_speed = np.asarray(test_speed)
test_speed = linear_map(test_speed, min_speed, max_speed, 0, 1)

print(f'Min_speed: {min_speed}')
print(f'Max_speed: {max_speed}')
print(f'Loaded {len(lidar)} Training samples '
      f'---- {(len(lidar)/total_number_samples)*100:0.2f}% of overall')
print(f'Loaded {len(test_lidar)} Testing samples '
      f'---- {(len(test_lidar)/total_number_samples)*100:0.2f}% of overall\n')

assert len(lidar) == len(servo) == len(speed)
assert len(test_lidar) == len(test_servo) == len(test_speed)

#======================================================
# Split Dataset
#======================================================

print('Splitting Data into Train/Test')
train_data = np.concatenate(
    (servo[:, np.newaxis], speed[:, np.newaxis]), axis=1)
test_data = np.concatenate(
    (test_servo[:, np.newaxis], test_speed[:, np.newaxis]), axis=1)
print(f'Train Data(lidar): {lidar.shape}')
print(f'Train Data(servo, speed): {servo.shape}, {speed.shape}')
print(f'Test Data(lidar): {test_lidar.shape}')
print(f'Test Data(servo, speed): {test_servo.shape}, {test_speed.shape}')

#======================================================
# Build Model
#======================================================

num_lidar_range_values = len(lidar[0])
print(f'num_lidar_range_values: {num_lidar_range_values}')
print(f'Resolution scales: {RESOLUTION_SCALES}')
for s in RESOLUTION_SCALES:
    print(f'  Scale {s:.2f}x -> {int(round(num_lidar_range_values * s))} input points')

# Build with placeholder native_length=1; patched after build
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

# Build and patch BN expected lengths from actual conv configs
model(np.zeros((1, num_lidar_range_values, 1), dtype=np.float32))
bn_native_lengths = patch_bn_native_lengths(model, num_lidar_range_values)
print("BN layer native lengths:", bn_native_lengths)
print(model.summary())

#======================================================
# Training
#======================================================

optimizer = Adam(lr)
loss_fn = tf.keras.losses.Huber()

history_train_loss = []
history_val_loss = []
history_val_per_res = defaultdict(list)

start_time = time.time()

for epoch in range(num_epochs):
    perm = np.random.permutation(len(lidar))
    lidar_shuffled = lidar[perm]
    train_data_shuffled = train_data[perm]

    epoch_loss = 0.0
    num_batches = 0

    if epoch == 0 and BN_DEBUG:
        print("\n[BN DEBUG] First batch trace:")

    for start in range(0, len(lidar_shuffled) - batch_size + 1, batch_size):
        lidar_batch = lidar_shuffled[start:start + batch_size]
        label_batch = train_data_shuffled[start:start + batch_size]
        label_tensor = tf.constant(label_batch, dtype=tf.float32)

        show_debug = BN_DEBUG and epoch == 0 and num_batches == 0

        with tf.GradientTape() as tape:
            total_loss = 0.0
            for scale_idx, scale in enumerate(RESOLUTION_SCALES):
                set_model_resolution(model, scale_idx)
                resized = resize_lidar_1d(lidar_batch, scale)
                resized = np.expand_dims(resized, -1).astype(np.float32)

                if show_debug:
                    print(f"\n  Scale {scale}x (bank {scale_idx}), "
                          f"input shape {resized.shape}:")

                preds = model(resized, training=True)
                total_loss += tf.reduce_mean(loss_fn(label_tensor, preds))

            avg_loss = total_loss / len(RESOLUTION_SCALES)

        gradients = tape.gradient(avg_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if show_debug:
            BN_DEBUG = False

        epoch_loss += float(avg_loss)
        num_batches += 1

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
        val_preds = np.concatenate(val_preds, axis=0)
        val_losses[scale] = float(tf.reduce_mean(
            loss_fn(test_data[:len(val_preds)], val_preds)
        ))

    avg_val_loss = sum(val_losses.values()) / len(val_losses)

    history_train_loss.append(avg_epoch_loss)
    history_val_loss.append(avg_val_loss)
    for scale in RESOLUTION_SCALES:
        history_val_per_res[scale].append(val_losses[scale])

    scale_str = " | ".join(
        f"{s:.2f}x: {val_losses[s]:.4f}" for s in RESOLUTION_SCALES)
    print(
        f'Epoch {epoch+1}/{num_epochs} - loss: {avg_epoch_loss:.4f} '
        f'- val_loss: {avg_val_loss:.4f} [{scale_str}]'
    )

print(f'=============>{int(time.time() - start_time)} seconds<=============')

#======================================================
# BN Calibration
#======================================================

if CALIBRATE_BN:
    print("Running BN calibration sweep...")
    calibrate_batch_norm(
        model, lidar, RESOLUTION_SCALES, batch_size, CALIBRATION_MAX_BATCHES)

#======================================================
# Loss Plot
#======================================================

plt.plot(history_train_loss, label='Train')
plt.plot(history_val_loss, label='Val (avg)', linewidth=2)
for scale in RESOLUTION_SCALES:
    plt.plot(
        history_val_per_res[scale],
        label=f'Val {scale:.2f}x',
        linestyle='--',
    )
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(loss_figure_path)
plt.close()

#======================================================
# Model Evaluation
#======================================================

print("==========================================")
print("Model Evaluation")
print("==========================================")

for scale_idx, scale in enumerate(RESOLUTION_SCALES):
    set_model_resolution(model, scale_idx)
    resized_test = resize_lidar_1d(test_lidar, scale)
    resized_test_input = np.expand_dims(resized_test, -1).astype(np.float32)

    preds = model(resized_test_input, training=False).numpy()
    hl = huber_loss(test_data, preds)
    speed_hl = huber_loss(test_data[:, 1], preds[:, 1])
    servo_hl = huber_loss(test_data[:, 0], preds[:, 0])

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
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"{model_name}_noquantized.tflite saved.")

rep_32 = lidar[:200].astype(np.float32)
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
    y_pred, inf_times = evaluate_model(m_name, test_lidar, test_data)
    all_inference_times_micros.append(inf_times)
    print(f'Huber Loss for {m_name}: {huber_loss(test_data, y_pred)}\n')

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