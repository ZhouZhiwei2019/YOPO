import os
import time
import threading
import psutil
import GPUtil
from torch.utils.tensorboard import SummaryWriter

class ResourceMonitor(threading.Thread):
    def __init__(self, log_dir="run/resource_monitor", interval=1.0):
        super().__init__()
        os.makedirs(log_dir, exist_ok=True) 
        self.writer = SummaryWriter(log_dir=log_dir)
        self.interval = interval
        self.keep_running = True
        self.start_time = time.time()

    def run(self):
        while self.keep_running:
            elapsed_time = time.time() - self.start_time
            self.log_resource_usage(elapsed_time)
            time.sleep(self.interval)

    def log_resource_usage(self, step):
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            self.writer.add_scalar("SystemMonitor/GPU_Usage(%)", gpu.load * 100, step)
            self.writer.add_scalar("SystemMonitor/GPU_Memory_Used_MB", gpu.memoryUsed, step)

        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_used = mem.used / (1024 ** 3)

        self.writer.add_scalar("SystemMonitor/CPU_Usage", cpu_percent, step)
        self.writer.add_scalar("SystemMonitor/RAM_Used_GB", mem_used, step)

    def stop(self):
        self.keep_running = False
        self.writer.close()
