import time
import numpy as np
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf


class TinyLidarNet(BasePlanner):
    def __init__(self, test_id, skip_n, pre, model_path,
                 scale=1.0, resolution_scales=None):
        """
        Args:
          resolution_scales: the list of scales the model was trained on,
            in the same order used during training. For a model trained at
            a single scale, pass [that_scale]. For the [1.0, 0.75] multires
            model, pass [1.0, 0.75]. `scale` must be one of these.
        """
        super().__init__("TinyLidarNet", test_id)
        self.pre = pre
        self.skip_n = skip_n
        self.model_path = model_path
        self.name = 'TinyLidarNet'

        if resolution_scales is None:
            raise ValueError(
                "resolution_scales must be provided and must match the list "
                "used to train the model at model_path"
            )
        self.resolution_scales = [float(s) for s in resolution_scales]
        self.scale = float(scale)
        self.scale_index = self._resolve_scale_index(
            self.scale, self.resolution_scales)
        self._scale_idx_tensor = np.array([self.scale_index], dtype=np.int32)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.input_details = self.interpreter.get_input_details()
        self.lidar_input_index, self.scale_input_index = \
            self._find_inputs(self.input_details)
        self.output_details = self.interpreter.get_output_details()

        self.scan_buffer = np.zeros((2, 20))
        self.temp_scan = []
        self._current_input_len = None

    @staticmethod
    def _resolve_scale_index(scale, resolution_scales):
        for i, s in enumerate(resolution_scales):
            if abs(s - scale) < 1e-6:
                return i
        raise ValueError(
            f"scale={scale} not in resolution_scales={resolution_scales} "
            f"for this model. Pass a scale the model was trained on."
        )

    @staticmethod
    def _find_inputs(input_details):
        lidar_idx = scale_idx = None
        for d in input_details:
            if d['dtype'] == np.int32:
                scale_idx = d['index']
            else:
                lidar_idx = d['index']
        if lidar_idx is None or scale_idx is None:
            raise RuntimeError(
                "TFLite model must expose one float lidar input "
                "and one int32 scale_index input"
            )
        return lidar_idx, scale_idx

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
                self.lidar_input_index, list(scan_input.shape))
            self.interpreter.allocate_tensors()
            self._current_input_len = scan_len

    def _set_inputs_and_invoke(self, scan_input):
        self._ensure_input_shape(scan_input)
        self.interpreter.set_tensor(self.lidar_input_index, scan_input)
        self.interpreter.set_tensor(self.scale_input_index, self._scale_idx_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def transform_obs(self, scan):
        scan = scan[:1080]
        scan = scan[::54]
        if self.scan_buffer.all() == 0:
            for i in range(self.scan_buffer.shape[0]):
                self.scan_buffer[i, :] = scan
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan
        return np.reshape(self.scan_buffer, (-1))

    def plan(self, obs):
        scans = obs['scan']

        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise

        chunks = [scans[i:i + 4] for i in range(0, len(scans), 4)]
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

            start_time = time.time()
            output = self._set_inputs_and_invoke(scans)
            inf_time = (time.time() - start_time) * 1000

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

            start_time = time.time()
            output = self._set_inputs_and_invoke(scans)
            inf_time = (time.time() - start_time) * 1000

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

            start_time = time.time()
            output = self._set_inputs_and_invoke(scans)
            inf_time = (time.time() - start_time) * 1000

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

            start_time = time.time()
            output = self._set_inputs_and_invoke(scans)
            inf_time = (time.time() - start_time) * 1000

            steer = output[0, 0]
            speed = output[0, 1]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed)
            action = np.array([steer, speed])

        return action
