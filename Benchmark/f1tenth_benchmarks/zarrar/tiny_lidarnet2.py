import time
import numpy as np
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf

BN_DEBUG = True


def bn_trace_from_interpreter(interpreter, scale, resolution_scales=None):
    """Print conv/BN spatial dims and which bank was selected."""
    if resolution_scales is None:
        resolution_scales = [1.0, 0.75]

    details = interpreter.get_tensor_details()
    print(f"\n[BN TRACE] scale={scale}")

    # Collect BN output spatial dims
    bn_spatials = {}
    for d in details:
        try:
            tensor = interpreter.get_tensor(d['index'])
        except ValueError:
            continue
        if len(tensor.shape) != 3:
            continue
        name = d['name']
        if '/add_1' in name:
            layer = name.split('/')[1] if '/' in name else name
            bn_spatials[layer] = tensor.shape[1]

    # Read expected_lengths from the GatherV2 constant tensors
    # or derive from the ratio of spatial dims across scales.
    # Simplest: compute expected from native = spatial / scale
    for layer, spatial in bn_spatials.items():
        native = int(round(spatial / scale))
        expected = [int(round(native * s)) for s in resolution_scales]
        diffs = [abs(e - spatial) for e in expected]
        selected = diffs.index(min(diffs))
        print(
            f"  {layer}: spatial={spatial} | "
            f"expected={expected} | "
            f"bank={selected} (scale={resolution_scales[selected]})"
        )


class TinyLidarNet(BasePlanner):
    def __init__(self, test_id, skip_n, pre, model_path, scale=1.0):
        super().__init__("TinyLidarNet", test_id)
        self.pre = pre
        self.skip_n = skip_n
        self.model_path = model_path
        self.name = 'TinyLidarNet'
        self.scale = float(scale)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.input_details = self.interpreter.get_input_details()
        self.input_index = self.input_details[0]["index"]
        self.output_details = self.interpreter.get_output_details()
        self.scan_buffer = np.zeros((2, 20))
        self.temp_scan = []
        self._current_input_len = None
        self._bn_traced = False

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def render_waypoints(self, *args, **kwargs):
        pass

    def resize_lidar_1d(self, scan):
        if abs(self.scale - 1.0) < 1e-6:
            return scan
        original_len = len(scan)
        new_len = int(round(original_len * self.scale))
        if new_len == original_len:
            return scan
        x_old = np.linspace(0, 1, original_len)
        x_new = np.linspace(0, 1, new_len)
        return np.interp(x_new, x_old, scan).astype(np.float32)

    def _ensure_input_shape(self, scan_input):
        scan_len = scan_input.shape[1]
        if scan_len != self._current_input_len:
            self.interpreter.resize_tensor_input(
                self.input_index, list(scan_input.shape))
            self.interpreter.allocate_tensors()
            self._current_input_len = scan_len

    def transform_obs(self, scan):
        self.scan_buffer
        scan = scan[:1080]
        scan = scan[::54]
        if self.scan_buffer.all() == 0:
            for i in range(self.scan_buffer.shape[0]):
                self.scan_buffer[i, :] = scan
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan
        scans = np.reshape(self.scan_buffer, (-1))
        return scans

    def plan(self, obs):
        scans = obs['scan']

        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise

        chunks = [scans[i:i+4] for i in range(0, len(scans), 4)]
        if self.pre == 1:
            scans = [np.mean(chunk) for chunk in chunks]
        elif self.pre == 2:
            scans = [np.max(chunk) for chunk in chunks]
        elif self.pre == 3:
            scans = [np.min(chunk) for chunk in chunks]
        elif self.pre == 4:
            scans = self.transform_obs(scans)
        else:
            scans = scans[::self.skip_n]

        if self.pre < 4:
            scans = np.array(scans)
            scans[scans > 10] = 10

            scans = self.resize_lidar_1d(scans)

            scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            scans = np.expand_dims(scans, axis=0)

            self._ensure_input_shape(scans)
            self.interpreter.set_tensor(self.input_index, scans)

            start_time = time.time()
            self.interpreter.invoke()
            inf_time = (time.time() - start_time) * 1000

            if BN_DEBUG and not self._bn_traced:
                bn_trace_from_interpreter(self.interpreter, self.scale)
                self._bn_traced = True

            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0, 0]
            speed = output[0, 1]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed)
            action = np.array([steer, speed])

        elif self.pre == 5:
            scans = np.array(scans)
            scans[scans > 10] = 10

            if len(self.temp_scan) < 1:
                self.temp_scan.append(scans)
                return np.array([0, 2])
            self.temp_scan.append(scans)
            scans = np.array(self.temp_scan)
            scans = np.expand_dims(scans, axis=0).astype(np.float32)
            scans = np.transpose(scans, (0, 2, 1))

            self._ensure_input_shape(scans)
            self.interpreter.set_tensor(self.input_index, scans)

            start_time = time.time()
            self.interpreter.invoke()
            inf_time = (time.time() - start_time) * 1000

            if BN_DEBUG and not self._bn_traced:
                bn_trace_from_interpreter(self.interpreter, self.scale)
                self._bn_traced = True

            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0, 0]
            speed = output[0, 1]
            self.temp_scan = self.temp_scan[1:]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed)
            action = np.array([steer, speed])

        elif self.pre == 6:
            scans = np.array(scans)
            scans[scans > 10] = 10

            if len(self.temp_scan) < 3:
                self.temp_scan.append(scans)
                return np.array([0, 2])
            self.temp_scan.append(scans)
            scans = np.array(self.temp_scan)
            scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            scans = np.expand_dims(scans, axis=0).astype(np.float32)

            self._ensure_input_shape(scans)
            self.interpreter.set_tensor(self.input_index, scans)

            start_time = time.time()
            self.interpreter.invoke()
            inf_time = (time.time() - start_time) * 1000

            if BN_DEBUG and not self._bn_traced:
                bn_trace_from_interpreter(self.interpreter, self.scale)
                self._bn_traced = True

            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0, 0]
            speed = output[0, 1]
            self.temp_scan = self.temp_scan[1:]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed)
            action = np.array([steer, speed])

        else:
            scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            scans = np.expand_dims(scans, axis=0)
            scans[scans > 10] = 10

            self._ensure_input_shape(scans)
            self.interpreter.set_tensor(self.input_index, scans)

            start_time = time.time()
            self.interpreter.invoke()
            inf_time = (time.time() - start_time) * 1000

            if BN_DEBUG and not self._bn_traced:
                bn_trace_from_interpreter(self.interpreter, self.scale)
                self._bn_traced = True

            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0, 0]
            speed = output[0, 1]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed)
            action = np.array([steer, speed])

        return action