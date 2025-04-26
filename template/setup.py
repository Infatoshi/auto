from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='cuda_benchmarks',
    ext_modules=[CUDAExtension(
        'template',
        sources=['bindings.cu', 'kernels/vector_addition.cu', 'kernels/relu.cu'],  # Updated sources list
        language='c++',  # Specify C++ for the extension
    )],
    cmdclass={'build_ext': BuildExtension},
)

