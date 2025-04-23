- Limitations on what it can modify.
- What is optimized with torch?
	- torch.compile
	- FA
	- triton
	- fp8 transformer engine
	- tf32 matmul
	- no_grad
- /mnt/d/CUDA/conv/
	├── main.py               # Entry point: runs profiler, verifier, and tests
	├── vector_addition.cu    # Kernel 1: Example CUDA kernel
	├── relu.cu              # Kernel 2: Example CUDA kernel
	├── bindings.cu          # CUDA bindings for raw kernels/cuDNN/cuBLAS (all seperate files to reduce bloat)
	├── setup.py             # python setup.py install
	├── kernels/             # Generated kernel files
	├── temp/                # Temporary files for testing new kernels
	├── configs/             # Configuration files for autotuning (batch sizes, tile sizes, etc.)
	└── logs/                # Debugging and profiling outputs
- latency OR throughput?
	- TTFT
	- batchsize
	- toks/sec
- need to make the architecture modular in case we want to add a kernel into the mix
- steps
	- init (setup/template files sort of like npm init)
	- debugging
		- macros
		- cuda-gdb (automated somehow?)
		- cout printing
	- optimization
		- nsys
		- ncu
		- streams
			- data
			- compute
		- row vs col major
		- inspiration from simon's and pranjalssh kernels
	- autotuning
		- grid search through:
			- batchsizes
			- tilesizes (thread level, warp level (wmma), warp group level (wgmma), block level)
			- all kernel variants for each problem  (low batchsize but long rows VS high batchsize and small rows for say, softmax)
			- grid/block dims -> how many threads/warps/blocks are issued to the kernel, considering launch overhead and such
- nn architecture
	- yes, inference only
	- N layers
	- non-fused ops
	- fused ops
- precision
	- fp16
	- bf16
	- fp32
	- int8
	- int4
	- accumulator precision
- GPU arch w/ deviceQuery
	- tensor core instructions available to us affecting warp/threads/registers required
- what could these kernel calls look like?
	- raw kernels
	- cuDNN fused ops
	- cuDNN non-fused
	- cuBLAS non-fused
	- cuBLAS-LT
	- cutlass
	- tensorRT
	- onnx runtime
- all the way down, square or rectangles? i know you can get correct output if you launch a kernel as a square, but then get 0s or NaNs or ever errors when you change to rectangle
- make everything as static as possible, aiming to optimize perf of a static model, not just a weird fused operation.
- use an agentic framework and jinja2 to manage a bunch of this
- could also aim for triton kernels to start?
- the goal of fusion is to minimize global memory transfers. we want to isolate and keep chunks of outputs on shared memory so we dont waste hundreds of clock cycles
- goal?
	- out = model.forward(x)
	- where "out.shape" might be (B, T, vocab_size) for training (NOT doing this) or (1, vocab_size) for inference
- write in C (basics) or C++ (FA, cutlass, pytorch)
