"""
Matrix Multiplication Kernels using CUTLASS DSL
High-performance GEMM operations with PyTorch integration
"""

import numpy as np
import torch
import cutlass
from typing import Optional, Tuple, Union

class GemmKernel:
    """
    High-performance matrix multiplication kernel using CUTLASS DSL
    Supports various data types and optimization strategies
    """
    
    def __init__(self, 
                 dtype: str = "float16",
                 layout: str = "row_major",
                 trans_a: bool = False,
                 trans_b: bool = False,
                 alpha: float = 1.0,
                 beta: float = 0.0):
        """
        Initialize GEMM kernel with specified configuration
        
        Args:
            dtype: Data type ('float16', 'float32', 'bfloat16')
            layout: Matrix layout ('row_major', 'column_major')
            trans_a: Whether to transpose matrix A
            trans_b: Whether to transpose matrix B
            alpha: Scaling factor for A*B
            beta: Scaling factor for C
        """
        self.dtype = dtype
        self.layout = layout
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.alpha = alpha
        self.beta = beta
        
        # Map string types to CUTLASS types
        self.dtype_map = {
            'float16': np.float16,
            'float32': np.float32,
            'bfloat16': cutlass.DataType.bf16
        }
        
    def _get_cutlass_dtype(self, dtype_str: str):
        """Convert string dtype to CUTLASS data type"""
        dtype_map = {
            'float16': cutlass.DataType.f16,
            'float32': cutlass.DataType.f32,
            'bfloat16': cutlass.DataType.bf16
        }
        return dtype_map.get(dtype_str, cutlass.DataType.f16)
    
    def _get_cutlass_layout(self, layout_str: str):
        """Convert string layout to CUTLASS layout"""
        layout_map = {
            'row_major': cutlass.LayoutType.RowMajor,
            'column_major': cutlass.LayoutType.ColumnMajor
        }
        return layout_map.get(layout_str, cutlass.LayoutType.RowMajor)
    
    def compile(self, 
                m: int, 
                n: int, 
                k: int,
                batch_size: int = 1) -> 'CompiledGemm':
        """
        Compile the GEMM kernel for specific dimensions
        
        Args:
            m: Number of rows in A and C
            n: Number of columns in B and C
            k: Number of columns in A and rows in B
            batch_size: Batch size for batched GEMM
            
        Returns:
            CompiledGemm instance ready for execution
        """
        return CompiledGemm(
            self, m, n, k, batch_size
        )
    
    def benchmark_against_pytorch(self,
                                  a: torch.Tensor,
                                  b: torch.Tensor,
                                  c: Optional[torch.Tensor] = None) -> dict:
        """
        Benchmark CUTLASS kernel against PyTorch
        
        Args:
            a: Input matrix A
            b: Input matrix B
            c: Optional input matrix C
            
        Returns:
            Dictionary with timing and accuracy metrics
        """
        # Ensure correct device and dtype
        device = a.device
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        
        if c is not None:
            c_np = c.detach().cpu().numpy()
        else:
            c_np = None
            
        # Compile and run CUTLASS kernel
        m, k = a.shape
        k_b, n = b.shape
        
        compiled = self.compile(m, n, k)
        result_cutlass = compiled.run(a_np, b_np, c_np)
        
        # Run PyTorch version
        torch_a = a.to(device)
        torch_b = b.to(device)
        
        if self.trans_a:
            torch_a = torch_a.t()
        if self.trans_b:
            torch_b = torch_b.t()
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result_torch = self.alpha * torch.mm(torch_a, torch_b)
        if c is not None:
            result_torch += self.beta * c.to(device)
        end.record()
        
        torch.cuda.synchronize()
        pytorch_time = start.elapsed_time(end)
        
        # Compare results
        result_torch_np = result_torch.detach().cpu().numpy()
        max_error = np.max(np.abs(result_cutlass - result_torch_np))
        mse = np.mean((result_cutlass - result_torch_np) ** 2)
        
        return {
            'cutlass_time': compiled.last_runtime,
            'pytorch_time': pytorch_time,
            'max_error': max_error,
            'mse': mse,
            'speedup': pytorch_time / compiled.last_runtime if compiled.last_runtime > 0 else float('inf')
        }


class CompiledGemm:
    """Compiled GEMM kernel ready for execution"""
    
    def __init__(self, kernel: GemmKernel, m: int, n: int, k: int, batch_size: int = 1):
        self.kernel = kernel
        self.m = m
        self.n = n
        self.k = k
        self.batch_size = batch_size
        
        # Create CUTLASS plan
        self.plan = cutlass.op.Gemm(
            element=self.kernel._get_cutlass_dtype(kernel.dtype),
            layout=self.kernel._get_cutlass_layout(kernel.layout)
        )
        
        self.last_runtime = None
        
    def run(self, 
            a: np.ndarray, 
            b: np.ndarray, 
            c: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute the compiled GEMM kernel
        
        Args:
            a: Input matrix A [m, k]
            b: Input matrix B [k, n]
            c: Optional input matrix C [m, n]
            
        Returns:
            Output matrix [m, n]
        """
        # Ensure correct shapes
        assert a.shape == (self.m, self.k), f"A shape {a.shape} != ({self.m}, {self.k})"
        assert b.shape == (self.k, self.n), f"B shape {b.shape} != ({self.k}, {self.n})"
        
        if c is not None:
            assert c.shape == (self.m, self.n), f"C shape {c.shape} != ({self.m}, {self.n})"
        
        # Create output tensor
        output = np.zeros((self.m, self.n), dtype=a.dtype)
        
        # Run CUTLASS kernel
        import time
        start_time = time.time()
        
        if c is not None:
            self.plan.run(a, b, c, output, 
                         alpha=self.kernel.alpha, beta=self.kernel.beta)
        else:
            self.plan.run(a, b, output, 
                         alpha=self.kernel.alpha, beta=self.kernel.beta)
        
        end_time = time.time()
        self.last_runtime = (end_time - start_time) * 1000  # Convert to ms
        
        return output


# Convenience functions for common use cases
def gemm_fp16(a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """FP16 GEMM with default settings"""
    kernel = GemmKernel(dtype='float16')
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    c_np = c.detach().cpu().numpy() if c is not None else None
    
    m, k = a.shape
    k_b, n = b.shape
    
    compiled = kernel.compile(m, n, k)
    result = compiled.run(a_np, b_np, c_np)
    
    return torch.from_numpy(result).to(a.device)


def gemm_fp32(a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """FP32 GEMM with default settings"""
    kernel = GemmKernel(dtype='float32')
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    c_np = c.detach().cpu().numpy() if c is not None else None
    
    m, k = a.shape
    k_b, n = b.shape
    
    compiled = kernel.compile(m, n, k)
    result = compiled.run(a_np, b_np, c_np)
    
    return torch.from_numpy(result).to(a.device)