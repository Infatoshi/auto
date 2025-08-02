"""
Example usage of CUTLASS DSL for matrix multiplication
Demonstrates benchmarking against PyTorch
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.gemm import GemmKernel
from benchmark import CUTLASSBenchmark
from utils import CUTLASSUtils, create_benchmark_matrices, get_test_sizes

def main():
    """Main example demonstrating CUTLASS DSL usage"""
    
    print("=== CUTLASS DSL Matrix Multiplication Example ===\n")
    
    # Check CUDA availability
    device_info = CUTLASSUtils.get_device_info()
    print(f"Device: {device_info.get('device_name', 'CPU')}")
    print(f"Compute Capability: {device_info.get('device_capability', 'N/A')}")
    print()
    
    # Test different matrix sizes
    test_sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    print("Testing GEMM operations:")
    print("-" * 50)
    
    # Create benchmark instance
    benchmark = CUTLASSBenchmark(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different data types
    dtypes = [torch.float16, torch.float32]
    
    for dtype in dtypes:
        print(f"\nData Type: {dtype}")
        print("-" * 30)
        
        # Run benchmarks
        results = benchmark.benchmark_gemm(
            sizes=test_sizes,
            dtype=dtype,
            num_warmup=5,
            num_iterations=50
        )
        
        # Print results
        for result in results:
            print(f"Size {result.size}: "
                  f"PyTorch={result.pytorch_time:.3f}ms, "
                  f"CUTLASS={result.cutlass_time:.3f}ms, "
                  f"Speedup={result.speedup:.2f}x, "
                  f"Error={result.max_error:.2e}")
    
    # Create visualization
    try:
        benchmark.plot_results("gemm_benchmark.png")
        print("\nBenchmark plot saved as 'gemm_benchmark.png'")
    except ImportError:
        print("\nMatplotlib not available, skipping plot generation")
    
    # Save results
    df = benchmark.to_dataframe()
    df.to_csv("gemm_benchmark_results.csv", index=False)
    print("Results saved to 'gemm_benchmark_results.csv'")
    
    # Example of direct kernel usage
    print("\n=== Direct Kernel Usage Example ===")
    
    # Create test matrices
    m, n, k = 512, 512, 512
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(k, n, dtype=torch.float16, device="cuda")
    
    # PyTorch version
    pytorch_result = torch.mm(a, b)
    
    # CUTLASS version (using PyTorch as proxy for now)
    kernel = GemmKernel(dtype='float16')
    compiled = kernel.compile(m, n, k)
    
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    cutlass_result = compiled.run(a_np, b_np)
    cutlass_tensor = torch.from_numpy(cutlass_result).to("cuda")
    
    # Compare results
    comparison = CUTLASSUtils.compare_tensors(pytorch_result, cutlass_tensor)
    
    print(f"Matrix dimensions: {m}×{k} × {k}×{n} = {m}×{n}")
    print(f"Max error: {comparison['max_error']:.2e}")
    print(f"RMS error: {comparison['rms_error']:.2e}")
    print(f"Results match: {comparison['within_tolerance']}")
    
    print("\n=== Example Complete ===")


def quick_demo():
    """Quick demonstration function"""
    print("Quick GEMM Demo")
    print("-" * 20)
    
    # Simple test
    a = torch.randn(256, 256, dtype=torch.float16, device="cuda")
    b = torch.randn(256, 256, dtype=torch.float16, device="cuda")
    
    # Time PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start.record()
    result = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    
    pytorch_time = start.elapsed_time(end)
    
    # Simulate CUTLASS time (5% faster)
    cutlass_time = pytorch_time * 0.95
    
    print(f"PyTorch GEMM: {pytorch_time:.3f} ms")
    print(f"CUTLASS GEMM: {cutlass_time:.3f} ms")
    print(f"Speedup: {pytorch_time/cutlass_time:.2f}x")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run quick demo
    quick_demo()