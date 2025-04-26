import time
import torch
import numpy as np
import template  # The module built from setup.py

def benchmark_function(func, *args, warmups=3, bench_runs=10):
    for _ in range(warmups):
        func(*args)  # Warmup runs
    times = []
    for _ in range(bench_runs):
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = np.mean(times)
    return result, avg_time  # Return the last result and average time

# Generate sample data
num_elements = 1024 * 1024  # 1 million elements for benchmarking
A_np = np.random.rand(num_elements).astype(np.float32)
B_np = np.random.rand(num_elements).astype(np.float32)
input_np = np.random.rand(num_elements).astype(np.float32)  # For ReLU

# Vector Addition Benchmark
print("Benchmarking Vector Addition:")

# Custom CUDA version
custom_result_add, custom_time_add = benchmark_function(template.vector_add, A_np, B_np)
print(f"Custom CUDA: Average time over 10 runs: {custom_time_add:.6f} seconds")

# PyTorch version
A_torch = torch.from_numpy(A_np).to('cuda')
B_torch = torch.from_numpy(B_np).to('cuda')
pytorch_result_add, pytorch_time_add = benchmark_function(lambda: torch.add(A_torch, B_torch).cpu().numpy(), warmups=3, bench_runs=10)
print(f"PyTorch: Average time over 10 runs: {pytorch_time_add:.6f} seconds")

# Compare results
if np.allclose(custom_result_add, pytorch_result_add, atol=1e-5):
    print("Vector Addition: Results match!")
else:
    print("Vector Addition: Results do not match.")

# ReLU Benchmark
print("\nBenchmarking ReLU:")

# Custom CUDA version
custom_result_relu, custom_time_relu = benchmark_function(template.relu, input_np)
print(f"Custom CUDA: Average time over 10 runs: {custom_time_relu:.6f} seconds")

# PyTorch version
input_torch = torch.from_numpy(input_np).to('cuda')
pytorch_result_relu, pytorch_time_relu = benchmark_function(lambda: torch.relu(input_torch).cpu().numpy(), warmups=3, bench_runs=10)
print(f"PyTorch: Average time over 10 runs: {pytorch_time_relu:.6f} seconds")

# Compare results
if np.allclose(custom_result_relu, pytorch_result_relu, atol=1e-5):
    print("ReLU: Results match!")
else:
    print("ReLU: Results do not match.")

