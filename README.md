# TODO
## prereqs
- `deviceQuery` (not `nvcc --version` or `nvidia-smi`)
- `ncu` and `nsys`
- pytorch w/ cuda support

## type of kernel
- activation
- conv
- gemm
- softmax
- norm
- pool
- element-wise
- reduction
- top-k
- sort

*start with passing a kernel into a config file and making an individual operation faster, rather than a a whole model at onc*