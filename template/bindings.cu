#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// Add these declarations for the kernels
extern __global__ void vectorAddKernel(float* A, float* B, float* C, int numElements);
extern __global__ void reluKernel(float* input, float* output, int numElements);

void vectorAddCUDA(float* d_A, float* d_B, float* d_C, int numElements, int threadsPerBlock) {
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaDeviceSynchronize();  // Ensure kernel finishes
}

void reluCUDA(float* d_input, float* d_output, int numElements, int threadsPerBlock) {
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements);
    cudaDeviceSynchronize();  // Ensure kernel finishes
}

PYBIND11_MODULE(template, m) {
    m.def("vector_add", [](py::array_t<float> A, py::array_t<float> B) {
        if (A.size() != B.size()) throw std::runtime_error("Arrays must be the same size");
        size_t numElements = A.size();
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, numElements * sizeof(float));
        cudaMalloc(&d_B, numElements * sizeof(float));
        cudaMalloc(&d_C, numElements * sizeof(float));
        
        // Copy data to device
        cudaMemcpy(d_A, A.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
        
        // Run kernel
        int threadsPerBlock = 256;  // Common choice
        vectorAddCUDA(d_A, d_B, d_C, numElements, threadsPerBlock);
        
        // Copy result back
        py::array_t<float> result(numElements);
        cudaMemcpy(result.mutable_data(), d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Free memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        
        return result;
    }, "Perform vector addition on GPU");

    m.def("relu", [](py::array_t<float> input) {
        size_t numElements = input.size();
        
        // Allocate device memory
        float *d_input, *d_output;
        cudaMalloc(&d_input, numElements * sizeof(float));
        cudaMalloc(&d_output, numElements * sizeof(float));
        
        // Copy data to device
        cudaMemcpy(d_input, input.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
        
        // Run kernel
        int threadsPerBlock = 256;
        reluCUDA(d_input, d_output, numElements, threadsPerBlock);
        
        // Copy result back
        py::array_t<float> result(numElements);
        cudaMemcpy(result.mutable_data(), d_output, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Free memory
        cudaFree(d_input);
        cudaFree(d_output);
        
        return result;
    }, "Apply ReLU on GPU");
}

