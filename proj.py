#!/usr/bin/env python
# coding: utf-8
import os, time, tracemalloc, csv
import numpy as np
import torch

# -----------------------------
# GPU Logging & Mean Computation
# -----------------------------
import csv
def write_result(label, n, t, mem, flops, category, method, avg):
    # Safely build the 4 GPU-average fields
    avg_fields = []
    if avg:
        for v in avg:
            if v is None:
                avg_fields.append("0")        # leave blank
            else:
                avg_fields.append(f"{v:.2f}")
    else:
        avg_fields = ["", "", "", ""]
        print("empty")

    # Build the full row
    row = [
        label,
        n,
        f"{t:.6f}",
        f"{mem:.2f}",
        f"{flops:.2e}",
        *avg_fields,
        category,
        method
    ]

    # Append it to results.csv
    with open('results.csv', mode='a', newline='') as f:
        csv.writer(f).writerow(row)


def compute_avg_gpu(csv_path='gpu_log.csv'):
    if not os.path.exists(csv_path): return None
    temps, powers, utils, mems = [], [], [], []
    with open(csv_path, mode='r') as f:
        for r in csv.DictReader(f):
            try:
                temps.append(float(r['temperature_C']))
                powers.append(float(r['power_W']))
                utils.append(float(r['gpu_util_percent']))
                mems.append(float(r['memory_used_MiB']))
            except: pass
    if not temps: return None
    return sum(temps)/len(temps), sum(powers)/len(powers), sum(utils)/len(utils), sum(mems)/len(mems)

# -----------------------------
# Matrix Multiplication Suite
# -----------------------------
def matmul_naive(A, B,n):
    tracemalloc.start()
    start = time.time()
    Cn = len(A)
    m = len(B[0])
    p = len(B)
    result = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end - start),(peak / 1024 / 1024),(2 * n**3 / (end - start))





def matmul_numpy (A_np, B_np,n):
    tracemalloc.start()
    start = time.time()
    C_numpy = np.matmul(A_np, B_np)
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (end - start),(peak / 1024 / 1024),(2 * n**3 / (end - start))




def matmul_pytorch(n):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_torch = torch.randn((n, n), device=device)
    B_torch = torch.randn((n, n), device=device)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    C_torch = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()
    
    return (end - start),(torch.cuda.memory_allocated() / 1024 ** 2),(2 * n**3 / (end - start))



def matmul(n):
    # Run each version separately with fresh GPU logs
    versions = [
        ("Naive_CPU_MatMul", lambda: matmul_naive(
            np.random.rand(min(n,400),min(n,400)).tolist(),
            np.random.rand(min(n,400),min(n,400)).tolist(),
            min(n,400)
        ), 'matmul', 'naive'),
        ("NumPy_CPU_MatMul", lambda: matmul_numpy(
            np.random.rand(n,n), np.random.rand(n,n), n
        ), 'matmul', 'numpy'),
        ("PyTorch_Gpu_MatMul", lambda: matmul_pytorch(n), 'matmul', 'pytorch')

    ]
    # Header for GPU log
    header = ['timestamp','temperature_C','gpu_util_percent','memory_used_MiB','power_W']
    for label, fn, category, method in versions:
        # clear gpu log
        with open('gpu_log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(header)
        # run benchmark
        t, mem, fl = fn()
        # scale naive time
        if 'Naive' in label:
            t *= n**3 / min(n,400)**3
            mem *= n**2 / min(n,400)**2
        # compute gpu averages
        avg = compute_avg_gpu() or (None, None, None, None)
        # write result row
        write_result(label, n, t, mem, fl, category, method, avg)
# -----------------------------
# Matrix Inversion Suite
# -----------------------------
def inverse_native(A_list):
    tracemalloc.start(); start=time.time()
    A=np.array(A_list,float); n=A.shape[0]
    I=np.eye(n); AI=np.hstack([A,I])
    for i in range(n):
        AI[i]/=AI[i,i]
        for j in range(n):
            if i!=j: AI[j]-=AI[i]*AI[j,i]
    end=time.time(); _, peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return end-start, peak/1024**2, 2*n**3/(end-start)

def inverse_numpy(A_np, n):
    tracemalloc.start(); start=time.time(); _=np.linalg.inv(A_np)
    end=time.time(); _, peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return end-start, peak/1024**2, 2*n**3/(end-start)

def inverse_pytorch(n):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A=torch.rand(n,n,device=device)
    torch.cuda.reset_peak_memory_stats(); start=time.time(); _=torch.inverse(A)
    if device.type=='cuda': torch.cuda.synchronize()
    end=time.time(); peak=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0
    return end-start, peak, 2*n**3/(end-start)

def inverse(n):
    # Run each version separately with fresh GPU logs
    versions = [
        ("Naive_CPU_Inv", lambda: inverse_native(
            np.random.rand(min(n,400),min(n,400)).tolist()
        ), 'inv', 'naive'),
        ("NumPy_CPU_Inv", lambda: inverse_numpy(
            np.random.rand(n,n), n
        ), 'inv', 'numpy'),
        ("PyTorch_Gpu_Inv", lambda: inverse_pytorch(n), 'inv', 'pytorch')
    ]
    header = ['timestamp','temperature_C','gpu_util_percent','memory_used_MiB','power_W']
    for label, fn, category, method in versions:
        # clear gpu log
        with open('gpu_log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(header)
        # benchmark
        t, mem, fl = fn()

        avg = compute_avg_gpu() or (None,None,None,None)
        # scale naive
        if 'Naive' in label:
            t *= n**3 / min(n,400)**3
            mem *= n**2 / min(n,400)**2
        write_result(label, n, t, mem, fl, category, method, avg)
# -----------------------------
# Pairwise Distance Suite
# -----------------------------
def dist_naive(X, Y):
    tracemalloc.start()
    start = time.time()
    N, D = len(X), len(X[0])
    M = len(Y)
    D_out = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            s = 0.0
            for d in range(D):
                diff = X[i][d] - Y[j][d]
                s += diff * diff
            D_out[i][j] = s ** 0.5
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (end - start),(peak / 1024 / 1024),(2 * N * M * D / (end - start))

    return (end - start),(peak / 1024 / 1024),(2 * N * M * D / (end - start))

def dist_numpy(X_np, Y_np, N,M,D):
    tracemalloc.start(); start=time.time()
    _=np.sqrt(np.sum(X_np**2,1)[:,None]+np.sum(Y_np**2,1)[None,:]-2*X_np.dot(Y_np.T))
    end=time.time(); _, peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return end-start, peak/1024**2, 2*N*M*D/(end-start)

def dist_pytorch(N,M,D):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X=torch.rand(N,D,device=device); Y=torch.rand(M,D,device=device)
    torch.cuda.reset_peak_memory_stats(); start=time.time(); _=torch.cdist(X,Y)
    if device.type=='cuda': torch.cuda.synchronize()
    end=time.time(); peak=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0
    return end-start, peak, 2*N*M*D/(end-start)

def distance(N, M, D):
    # Versions
    versions = [
        ("Naive_CPU_Dist", lambda: dist_naive(
            np.random.rand(min(N,400),min(D,400)).tolist(),
            np.random.rand(min(M,400),min(D,400)).tolist()
        ), 'dist', 'naive'),
        ("NumPy_CPU_Dist", lambda: dist_numpy(
            np.random.rand(N,D), np.random.rand(M,D), N, M, D
        ), 'dist', 'numpy'),
        ("PyTorch_Gpu_Dist", lambda: dist_pytorch(N, M, D), 'dist', 'pytorch')
    ]
    header = ['timestamp','temperature_C','gpu_util_percent','memory_used_MiB','power_W']
    for label, fn, category, method in versions:
        # clear gpu log
        with open('gpu_log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(header)
        # run
        t, mem, fl = fn()
        # scale naive
        avg = compute_avg_gpu() or (None,None,None,None)
        if 'Naive' in label:
            scale = (N*M*D) / (min(M,400) * min(N,400) * min(D,400))
            t *= scale
            mem *= ((N+M)*D) / ((min(M,400) + min(N,400)) * min(D,400))
        write_result(label, (N+M+D)//3, t, mem, fl, category, method, avg)
