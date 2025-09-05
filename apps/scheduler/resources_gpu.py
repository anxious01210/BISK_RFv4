# apps/scheduler/resources_gpu.py
import time

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def read_gpu_util(gpu_index: int):
    if not _NVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return int(util.gpu)  # 0..100
    except Exception:
        return None


class GpuPacer:
    """Keep average utilization near a target by light sleeping when above target."""

    def __init__(self, gpu_index: int, target_util: int, window_ms: int = 1500):
        self.idx = int(gpu_index)
        self.target = max(1, min(100, int(target_util)))
        self.window = max(250, int(window_ms))
        self.samples = []  # [(ts, util)]

    def _avg(self, now):
        cutoff = now - self.window / 1000.0
        self.samples = [(t, u) for (t, u) in self.samples if t >= cutoff]
        if not self.samples:
            return None
        return sum(u for _, u in self.samples) / len(self.samples)

    def maybe_sleep(self):
        util = read_gpu_util(self.idx)
        if util is None:
            return
        now = time.time()
        self.samples.append((now, util))
        avg = self._avg(now)
        if avg is not None and avg > self.target:
            over = avg - self.target
            # proportional backoff: cap at 150ms
            time.sleep(min(0.015 * over, 0.150))
