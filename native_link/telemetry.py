import psutil
import platform
import time
from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    NVMLError
)

class Monitor:
    """monitor system hardware and network performance"""
    def __init__(self):
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
        except NVMLError:
            self.gpu_handle = None
            
        self.prev_net = psutil.net_io_counters() # store previous network stats for speed calculation
        self.prev_time = time.time()

    def shell_config(self):
        """
        get operating system and version details
        
        returns:
            str: formatted string of system information
        """
        info = (
            f"OS: {platform.system()}\n"
            f"Release: {platform.release()}\n"
            f"Version: {platform.version()}\n"
            f"Details: {platform.platform()}"
        )
        return info

    def system_stats(self):
        """
        calculate current usage for cpu, ram, gpu, disk and network
        
        returns:
            tuple: strings containing usage percentages and speeds
        """
        cpu_usage = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # --- GPU ---
        if self.gpu_handle:
            try:
                gpu_util = nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                gpu_mem = nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_used = round(gpu_mem.used / 1e9, 2)
                gpu_mem_total = round(gpu_mem.total / 1e9, 2)
                gpu_stats = f"GPU: {gpu_mem_used}/{gpu_mem_total} GB"
            except NVMLError:
                gpu_stats = "GPU: N/A"
        else:
            gpu_stats = "GPU: N/A"

        # --- network---
        net = psutil.net_io_counters()
        current_time = time.time()
        
        time_diff = current_time - self.prev_time
        
        # total bytes transferred since last check
        total_bytes_now = net.bytes_sent + net.bytes_recv
        total_bytes_prev = self.prev_net.bytes_sent + self.prev_net.bytes_recv
        
        # zero division handling
        if time_diff > 0:
            total_speed = (total_bytes_now - total_bytes_prev) / time_diff / 1e6  # MB/s
        else:
            total_speed = 0.0

        # update state -> next call
        self.prev_net = net
        self.prev_time = current_time

        return (
            f"CPU: {cpu_usage}%", 
            f"RAM: {ram.percent}%",
            gpu_stats,
            f"NET: {total_speed:.2f} MB/s", 
            f"DISK: {disk.percent}%",
        )