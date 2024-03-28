//Joel Coghlin - 20228087
//Machine Problem 2
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>   // Include for rand() function
#include <random>

#define TILE_WIDTH 5

/*/ Kernel function to perform matrix multiplication on GPU
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
*/


__global__ void matrixMul(float* A, float* B, float* C, int width) {
    __shared__ float share_A[TILE_WIDTH * TILE_WIDTH];
    __shared__ float share_B[TILE_WIDTH * TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by* TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH +tx;

    float sum = 0.0;

    if (row >= width || col >= width) return;
    for (int m = 0; m < (width-1)/TILE_WIDTH + 1; ++m) {
        int A_index = row * width + m * TILE_WIDTH + tx;   //load the matrices into shared memory arrays.
        int B_index = (m * TILE_WIDTH + ty) * width + col;

        share_A[ty * TILE_WIDTH + tx] = A[A_index];
        share_B[tx * TILE_WIDTH + ty] = B[B_index];


        //check if the shared indexing of A tile is out of bounds (row matrix)
        /*if (ty * TILE_WIDTH + tx < width) {
            share_A[ty * TILE_WIDTH + tx] = A[A_index];
        }
        else {
            share_A[ty * TILE_WIDTH + tx] = 0.0;
        }

        //check if the shared indexing of the B tile is out of bounds (column matrix)
        if (tx * TILE_WIDTH + ty < width) {
            share_B[tx * TILE_WIDTH + ty] = B[B_index];
        }
        else {
            share_B[tx * TILE_WIDTH + ty] = 0.0;
        }
        */
        
        
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            if(m*TILE_WIDTH +k < width)
                sum += share_A[ty*TILE_WIDTH + k]*share_B[k*TILE_WIDTH + tx];           //do mat mul with these new arrays
        }
        __syncthreads();
    }

   
    C[row * width + col] = sum;
    
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
    int width = 10;
    int size = width * width * sizeof(float);

    // Allocate memory for host matrices
    float* h_A = 0;
    float* h_B = 0;
    float* h_C = 0;
    float* h_C_CPU = 0; // For CPU computation

    cudaMallocHost((void**)&h_A, size);
    cudaMallocHost((void**)&h_B, size);
    cudaMallocHost((void**)&h_C, size);
    cudaMallocHost((void**)&h_C_CPU, size);

    /// Seed the random number generator
    srand(time(nullptr));

    /*// Initialize host matrices with random numbers from 1 to 10
    for (int i = 0; i < width * width; i++) {
        h_A[i] = static_cast<float>(rand() % 10 + 1);  // Random number from 1 to 10
        h_B[i] = static_cast<float>(rand() % 10 + 1);  // Random number from 1 to 10
    }
    */

    for (int i = 0; i < width * width; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
    }

    for (int i = 0; i < width * width; i++) {
        h_B[i] = (float)rand() / RAND_MAX;
    }

    for (int i = 0; i < width * width; i++) {
        h_C[i] = 0;
        h_C_CPU[i] = 0;
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


    //Define grid and block dimensions
    dim3 dimGrid = dim3((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    dim3 dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);

    //Launch kernel 
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

    //Verify
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_C[i] - h_C_CPU[i]) > 1) {
            printf("Test FAILED\n");
            //break;
        }
    }

    for (int e = 0; e < width * width; e++) {
        printf(" %.3f\n", h_C[e]);
    }

    for (int w = 0; w < width * width; w++) {
        printf(" %.3f\n", h_C_CPU[w]);
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


    //Copy result matrix from device to host
    //cudaEvent_t start, stop;
    //Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_C_CPU);

    return 0;
}

