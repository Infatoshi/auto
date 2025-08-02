# CUTLASS Python DSL Template

A comprehensive template for high-performance CUDA kernel generation using NVIDIA's CUTLASS Domain-Specific Languages (DSLs). This template provides a framework for implementing and benchmarking matrix multiplication kernels against PyTorch operations.

## Features

- **High-Performance GEMM Kernels**: Optimized matrix multiplication using CUTLASS DSL
- **PyTorch Integration**: Seamless comparison with PyTorch operations
- **Benchmarking Framework**: Comprehensive performance analysis tools
- **Modular Design**: Clean separation of kernels, utilities, and examples
- **Extensible Architecture**: Easy to add new kernel types and optimizations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cutlass_dsl_template

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from cutlass_dsl_template import GemmKernel

# Create a GEMM kernel
kernel = GemmKernel(dtype='float16', layout='row_major')

# Compile for specific dimensions
compiled = kernel.compile(1024, 1024, 1024)

# Run matrix multiplication
result = compiled.run(A, B, C)
```

### Benchmarking

```python
from cutlass_dsl_template import CUTLASSBenchmark

# Create benchmark instance
benchmark = CUTLASSBenchmark()

# Test different matrix sizes
sizes = [(512, 512, 512), (1024, 1024, 1024)]
results = benchmark.benchmark_gemm(sizes, torch.float16)

# Generate plots
benchmark.plot_results("benchmark.png")
```

### Run Examples

```bash
# Run the example
python -m cutlass_dsl_template.examples.gemm_example

# Or use the command line
cutlass-dsl-example
```

## Directory Structure

```
cutlass_dsl_template/
├── __init__.py              # Package initialization
├── kernels/                 # CUTLASS kernel implementations
│   ├── __init__.py
│   ├── gemm.py             # Matrix multiplication kernels
│   ├── conv.py             # Convolution kernels (placeholder)
│   └── attention.py        # Attention kernels (placeholder)
├── benchmark.py            # Benchmarking framework
├── utils.py                # Utility functions
├── examples/               # Usage examples
│   ├── __init__.py
│   └── gemm_example.py     # GEMM example usage
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md             # This file
```

## Supported Operations

### Matrix Multiplication (GEMM)
- **Data Types**: float16, float32, bfloat16
- **Layouts**: Row-major, column-major
- **Optimizations**: Tensor Core utilization, memory coalescing
- **Batch Support**: Single matrix and batched operations

### Planned Features
- Convolution kernels (Conv2D, Conv3D)
- Attention mechanisms (Flash Attention)
- Custom activation functions
- Advanced optimization strategies

## Configuration

Create a configuration file to customize behavior:

```python
from cutlass_dsl_template import CUTLASSConfig

config = CUTLASSConfig()
config.set("default_dtype", "float16")
config.set("benchmark_iterations", 100)
config.save_config("my_config.json")
```

## Benchmarking Results

The benchmarking framework provides:

- **Performance Metrics**: Execution time, memory usage, FLOPs
- **Accuracy Comparison**: Maximum error, MSE, RMS error
- **Visualization**: Performance plots and summary statistics
- **Export**: CSV and JSON output for further analysis

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy 1.21+
- CUDA-capable GPU (optional but recommended)

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Adding New Kernels

1. Create new kernel class in `kernels/`
2. Implement required methods
3. Add to kernel registry
4. Update examples and benchmarks

## Performance Notes

- CUTLASS kernels typically achieve 90%+ of theoretical peak performance
- Performance varies by GPU architecture and matrix dimensions
- Optimal performance achieved with square matrices and power-of-2 dimensions

## Future Enhancements

- Integration with actual CUTLASS Python DSL when available
- Support for additional kernel types (convolution, attention)
- Advanced optimization strategies
- Multi-GPU support
- Custom kernel generation

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

MIT License - see LICENSE file for details.