#!/usr/bin/env python3
import time
import numpy as np
from numba import njit
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf
import importlib.util
import sys
import os
tf.print = lambda *args, **kwargs: None

# ─────────────────────────────────────────────────────────────
# Dynamically import ConditionalBatchNorm from train3.py
# ─────────────────────────────────────────────────────────────
path_to_module = "../../../train3.py"
spec = importlib.util.spec_from_file_location("train3", path_to_module)
train3 = importlib.util.module_from_spec(spec)
sys.modules["train3"] = train3
spec.loader.exec_module(train3)
ConditionalBatchNorm = train3.ConditionalBatchNorm

# ─────────────────────────────────────────────────────────────
# TensorFlow environment setup
# ─────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU for small models
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"


# ─────────────────────────────────────────────────────────────
# TinyLidarNet planner
# ─────────────────────────────────────────────────────────────
class TinyLidarNet(BasePlanner):
    def __init__(self, test_id, skip_n, pre, model_path):
        super().__init__("TinyLidarNet", test_id)
        self.pre = pre
        self.skip_n = skip_n
        self.model_path = model_path
        self.name = 'TinyLidarNet'

        print(f"Loading TensorFlow model from: {model_path}")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"ConditionalBatchNorm": ConditionalBatchNorm}
        )
        print("Model loaded successfully.")

        self.scan_buffer = np.zeros((2, 20))
        self.temp_scan = []

    @tf.function
    def infer(self, scan, res_id):
        return self.model([scan, res_id], training=False)

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def plan(self, obs):
        scans = np.array(obs['scan'], dtype=np.float32)

        # === Downsample lidar exactly like in training ===
        if self.skip_n > 1:
            scans = scans[::self.skip_n]

        # 🔹 Original noise logic preserved exactly
        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        scans[scans > 10] = 10

        # Reshape for model input
        scans = np.expand_dims(scans, axis=-1).astype(np.float32)  # (N, 1)
        scans = np.expand_dims(scans, axis=0)  # (1, N, 1)

        # Determine resolution group (same logic as training)
        num_points = scans.shape[1]
        if num_points <= 360:
            res_id = 0  # low
        elif num_points <= 720:
            res_id = 1  # mid
        else:
            res_id = 2  # high
        # Must match Input(shape=(1,), dtype=int32)
        res_input = np.array([[res_id]], dtype=np.int32)  # (1, 1)

        # Run inference
        start_time = time.time()
        output = self.infer(scans, res_input).numpy()
        inf_time = (time.time() - start_time) * 1000.0  # ms

        steer = output[0, 0]
        speed = output[0, 1]
        speed = self.linear_map(speed, 0, 1, 1, 8)

        tf.print(f"Resolution ID:", res_id, "| Inference time (ms):", inf_time)
        return np.array([steer, speed])
