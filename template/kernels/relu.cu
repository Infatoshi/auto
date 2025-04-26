__global__ void reluKernel(float* input, float* output, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        output[i] = fmaxf(input[i], 0.0f);  // ReLU: max(0, x)
    }
} 
