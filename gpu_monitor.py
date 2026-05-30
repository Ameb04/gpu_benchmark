# gpu_monitor.py
import csv
import time
import signal
import sys
from datetime import datetime
import pynvml

csv_file = 'gpu_log.csv'
interval = 0.00001  # 10 ms sampling

def handler(signum, frame):
    # Gracefully shut down NVML and exit
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    sys.exit(0)

# Catch Ctrl+C even during init
signal.signal(signal.SIGINT, handler)

# 1) Initialize NVML
try:
    pynvml.nvmlInit()
except pynvml.NVMLError as err:
    print(f"NVML init error: {err}")
    sys.exit(1)

# Only monitoring GPU 0 here; extend as needed
device = pynvml.nvmlDeviceGetHandleByIndex(0)

# 2) Clear/create CSV and write header once
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp',
        'temperature_C',
        'gpu_util_percent',
        'memory_used_MiB',
        'power_W',
    ])

# 3) Take an initial sample immediately (guaranteed at least one row)
ts    = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
temp  = pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)
util  = pynvml.nvmlDeviceGetUtilizationRates(device).gpu
mem   = pynvml.nvmlDeviceGetMemoryInfo(device)
power = pynvml.nvmlDeviceGetPowerUsage(device) / 1000.0

with open(csv_file, mode='a', newline='') as f:
    csv.writer(f).writerow([ts, temp, util, mem.used//(1024**2), power])

# 4) Now enter the timed loop: sleep, then sample & append
while True:
    #time.sleep(interval)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    try:
        temp  = pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU)
        util  = pynvml.nvmlDeviceGetUtilizationRates(device).gpu
        mem   = pynvml.nvmlDeviceGetMemoryInfo(device)
        power = pynvml.nvmlDeviceGetPowerUsage(device) / 1000.0
        row   = [ts, temp, util, mem.used//(1024**2), power]
    except pynvml.NVMLError:
        # If NVML throws during runtime, record timestamp only
        row = [ts, None, None, None, None]

    with open(csv_file, mode='a', newline='') as f:
        csv.writer(f).writerow(row)

