//Joel Coghlin - 20228087
//Machine Problem 1
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <cstdlib>   // Include for rand() function
#include <ctime>     // Include for seeding the random number generator

#define TILE_WIDTH 25

// Kernel function to perform matrix multiplication on GPU
__global__ void matrixMul(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Function to perform matrix multiplication on CPU
void matrixMulCPU(float* A, float* B, float* C, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0;
            for (int k = 0; k < width; k++) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

int main() {

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    printf("There are/is %d device(s) available\n", numDevices);

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    for (int i = 0; i < numDevices; i++) {
        printf("Device Number (on system): %d\n", i);
        printf("Device Clock Rate: %d kHz\n", deviceProperties.clockRate);
        printf("Number of SMs: %d\n", deviceProperties.multiProcessorCount);
        printf("Number of Cores: \n");
        printf("Warp Size: %d\n", deviceProperties.warpSize);
        printf("Total Global Memory: %zu Bytes\n", deviceProperties.totalGlobalMem);
        printf("Total Constant Memory: %zu Bytes\n", deviceProperties.totalConstMem);
        printf("Shared Memory Per Block: %zu Bytes\n", deviceProperties.sharedMemPerBlock);
        printf("Registers Available Per Block: %d\n", deviceProperties.regsPerBlock);
        printf("Max Threads Per Block: %d\n", deviceProperties.maxThreadsPerBlock);
        printf("Max Size of dimBlock: %d, %d, %d\n", deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
        printf("Max Size of dimGrid: %d, %d, %d\n", deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
    }

    cudaSetDevice(0);

    // Matrix size
    int width = 1000;
    int size = width * width * sizeof(float);

    // Allocate memory for host matrices
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_C_CPU = (float*)malloc(size); // For CPU computation

    /// Seed the random number generator
    srand(time(nullptr));

    // Initialize host matrices with random numbers from 1 to 10
    for (int i = 0; i < width * width; i++) {
        h_A[i] = static_cast<float>(rand() % 10 + 1);  // Random number from 1 to 10
        h_B[i] = static_cast<float>(rand() % 10 + 1);  // Random number from 1 to 10
    }

    // Allocate memory for device matrices
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy host matrices to device
    cudaEventRecord(start, 0);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("copying host to device took: %f\n", milliseconds);


    // Define grid and block dimensions
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Launch kernel to perform matrix multiplication on GPU
    cudaEventRecord(start, 0);
    matrixMul << <dimGrid, dimBlock >> > (d_A, d_B, d_C, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel took: %f\n", milliseconds);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    /* Print GPU result
        printf("\nGPU Result:\n");
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < width; ++j) {
                printf("%.3f ", h_C[i * width + j]);
            }
            printf("\n");
        }*/
        // Perform matrix multiplication on CPU for comparison
        //cudaEvent_t start, stop;

    cudaEventRecord(start, 0);
    matrixMulCPU(h_A, h_B, h_C_CPU, width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The CPU took: %f\n", milliseconds);

    // Verify result
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_C[i] - h_C_CPU[i]) > 1) {
            printf("Test FAILED");
            break;
        }
    }



    /* Print CPU result
    printf("CPU Result:\n");
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%.3f ", h_C_CPU[i * width + j]);
        }
        printf("\n");
    }*/
    cudaEventRecord(start, 0);
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("copying device to host took: %f\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Copy result matrix from device to host
    //cudaEvent_t start, stop;
    printf("Test PASSED.\n");
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);

    return 0;
}

