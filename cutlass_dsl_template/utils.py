"""
Utility functions for CUTLASS DSL operations
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
import os
import json

class CUTLASSUtils:
    """Utility functions for CUTLASS DSL operations"""
    
    @staticmethod
    def create_test_matrices(m: int, n: int, k: int, 
                           dtype: torch.dtype = torch.float16,
                           device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create test matrices for GEMM operations"""
        device = torch.device(device)
        
        # Create random matrices
        a = torch.randn(m, k, dtype=dtype, device=device)
        b = torch.randn(k, n, dtype=dtype, device=device)
        c = torch.randn(m, n, dtype=dtype, device=device)
        
        return a, b, c
    
    @staticmethod
    def validate_inputs(a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None) -> bool:
        """Validate input tensors for GEMM operations"""
        if len(a.shape) != 2 or len(b.shape) != 2:
            return False
            
        if a.shape[1] != b.shape[0]:
            return False
            
        if c is not None:
            if len(c.shape) != 2:
                return False
            if a.shape[0] != c.shape[0] or b.shape[1] != c.shape[1]:
                return False
                
        return True
    
    @staticmethod
    def get_memory_usage(tensor: torch.Tensor) -> int:
        """Get memory usage of a tensor in bytes"""
        return tensor.numel() * tensor.element_size()
    
    @staticmethod
    def estimate_flops(m: int, n: int, k: int) -> int:
        """Estimate FLOPs for GEMM operation: 2 * m * n * k"""
        return 2 * m * n * k
    
    @staticmethod
    def get_device_info() -> dict:
        """Get information about the CUDA device"""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
            
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        return {
            "cuda_available": True,
            "device_name": props.name,
            "device_capability": f"{props.major}.{props.minor}",
            "total_memory": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "max_threads_per_block": props.max_threads_per_block,
            "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor
        }
    
    @staticmethod
    def save_results(results: list, filename: str, directory: str = "results") -> None:
        """Save benchmark results to JSON file"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    @staticmethod
    def load_results(filename: str, directory: str = "results") -> list:
        """Load benchmark results from JSON file"""
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    @staticmethod
    def format_time(time_ms: float) -> str:
        """Format time in human-readable format"""
        if time_ms < 1.0:
            return f"{time_ms*1000:.2f} μs"
        elif time_ms < 1000.0:
            return f"{time_ms:.2f} ms"
        else:
            return f"{time_ms/1000:.2f} s"
    
    @staticmethod
    def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                       tolerance: float = 1e-5) -> dict:
        """Compare two tensors and return accuracy metrics"""
        if tensor1.shape != tensor2.shape:
            return {"error": "Shape mismatch"}
            
        diff = torch.abs(tensor1 - tensor2)
        
        return {
            "max_error": torch.max(diff).item(),
            "mean_error": torch.mean(diff).item(),
            "rms_error": torch.sqrt(torch.mean(diff ** 2)).item(),
            "within_tolerance": torch.all(diff < tolerance).item(),
            "tolerance": tolerance
        }


class CUTLASSConfig:
    """Configuration management for CUTLASS DSL"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = {
            "default_dtype": "float16",
            "default_layout": "row_major",
            "benchmark_iterations": 100,
            "warmup_iterations": 10,
            "tolerance": 1e-5,
            "device": "cuda"
        }
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            self.config.update(config_data)
    
    def save_config(self, config_file: str) -> None:
        """Save configuration to JSON file"""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value) -> None:
        """Set configuration value"""
        self.config[key] = value


# Example usage utilities
def create_benchmark_matrices() -> dict:
    """Create standard benchmark matrices"""
    return {
        "small": (128, 128, 128),
        "medium": (512, 512, 512),
        "large": (1024, 1024, 1024),
        "xlarge": (2048, 2048, 2048)
    }


def get_test_sizes() -> list:
    """Get common test sizes for benchmarking"""
    return [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (64, 1024, 512),    # BERT-like
        (512, 1024, 64),   # ResNet-like
        (1024, 4096, 1024)  # Transformer-like
    ]


def setup_environment() -> dict:
    """Setup and return environment information"""
    info = CUTLASSUtils.get_device_info()
    
    if info["cuda_available"]:
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
    return info


# Convenience functions
def quick_test(m: int = 512, n: int = 512, k: int = 512, 
              dtype: torch.dtype = torch.float16) -> dict:
    """Quick test function for GEMM"""
    a, b, c = CUTLASSUtils.create_test_matrices(m, n, k, dtype)
    
    # Time PyTorch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start.record()
    result = torch.mm(a, b) + c
    end.record()
    torch.cuda.synchronize()
    
    pytorch_time = start.elapsed_time(end)
    
    return {
        "m": m, "n": n, "k": k,
        "dtype": str(dtype),
        "pytorch_time_ms": pytorch_time,
        "flops": CUTLASSUtils.estimate_flops(m, n, k),
        "memory_usage": CUTLASSUtils.get_memory_usage(a) + 
                       CUTLASSUtils.get_memory_usage(b) + 
                       CUTLASSUtils.get_memory_usage(c)
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing CUTLASS DSL utilities...")
    
    # Check CUDA availability
    device_info = setup_environment()
    print(f"Device info: {device_info}")
    
    # Quick test
    result = quick_test(256, 256, 256)
    print(f"Quick test result: {result}")
    
    print("Utilities test completed!")