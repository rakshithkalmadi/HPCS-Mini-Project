#include <stdio.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include <chrono>

using namespace cv;
using namespace std;

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__device__ int b(int a, int b, int c) {
    int positive_count = 0;

    if (a > 0) {
        positive_count++;
    }
    if (b > 0) {
        positive_count++;
    }
    if (c > 0) {
        positive_count++;
    }

    return (positive_count >= 2) ? 1 : 0;
}

__global__ void ltpcKernel(uchar* input, uchar* output, int rows, int cols) {
    __shared__ uchar sharedInput[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load block into shared memory
    int sharedX = threadIdx.x + 1;
    int sharedY = threadIdx.y + 1;
    if (x < cols && y < rows) {
        sharedInput[sharedY][sharedX] = input[y * cols + x];
    }

    // Load boundary pixels
    if (threadIdx.x == 0) {
        if (x > 0) {
            sharedInput[sharedY][0] = input[y * cols + x - 1];
        } else {
            sharedInput[sharedY][0] = 0;
        }
    }
    if (threadIdx.x == blockDim.x - 1) {
        if (x < cols - 1) {
            sharedInput[sharedY][BLOCK_SIZE_X + 1] = input[y * cols + x + 1];
        } else {
            sharedInput[sharedY][BLOCK_SIZE_X + 1] = 0;
        }
    }
    if (threadIdx.y == 0) {
        if (y > 0) {
            sharedInput[0][sharedX] = input[(y - 1) * cols + x];
        } else {
            sharedInput[0][sharedX] = 0;
        }
    }
    if (threadIdx.y == blockDim.y - 1) {
        if (y < rows - 1) {
            sharedInput[BLOCK_SIZE_Y + 1][sharedX] = input[(y + 1) * cols + x];
        } else {
            sharedInput[BLOCK_SIZE_Y + 1][sharedX] = 0;
        }
    }

    __syncthreads();

    if (x < cols && y < rows) {
        int C = sharedInput[sharedY][sharedX];
        int P0 = sharedInput[sharedY][sharedX + 1];
        int P1 = sharedInput[sharedY + 1][sharedX + 1];
        int P2 = sharedInput[sharedY + 1][sharedX];
        int P3 = sharedInput[sharedY + 1][sharedX - 1];
        int P4 = sharedInput[sharedY][sharedX - 1];
        int P5 = sharedInput[sharedY - 1][sharedX - 1];
        int P6 = sharedInput[sharedY - 1][sharedX];
        int P7 = sharedInput[sharedY - 1][sharedX + 1];

        int CP[8];
        CP[0] = b(P7 - C, P0 - C, P1 - C);
        CP[1] = b(P1 - C, P2 - C, P3 - C);
        CP[2] = b(P3 - C, P4 - C, P5 - C);
        CP[3] = b(P5 - C, P6 - C, P7 - C);
        CP[4] = b(P6 - P0, C - P0, P2 - P0);
        CP[5] = b(P4 - P2, C - P2, P0 - P2);
        CP[6] = b(P2 - P4, C - P4, P6 - P4);
        CP[7] = b(P0 - P6, C - P6, P4 - P6);

        int decimal_value = 0;
        for (int i = 7; i >= 0; i--) {
            decimal_value = decimal_value * 2 + CP[i];
        }

        output[y * cols + x] = static_cast<uchar>(decimal_value);
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    //create events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    Mat image = imread("image1.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Image not found or could not be opened.\n");
        return 1;
    }

    auto image_load_time = std::chrono::high_resolution_clock::now();
    auto image_loading_duration = std::chrono::duration_cast<std::chrono::milliseconds>(image_load_time - start_time);

    int rows = image.rows;
    int cols = image.cols;

    Mat result_image(rows, cols, CV_8UC1);

    uchar* d_input, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(uchar));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(uchar));

    cudaMemcpy(d_input, image.data, rows * cols * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(event1, 0);
    ltpcKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
    cudaEventRecord(event2, 0);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    float dt_ms;
    cudaEventElapsedTime(&dt_ms, event1, event2);

    cudaMemcpy(result_image.data, d_output, rows * cols * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    printf("Image loading time: %ld ms\n", image_loading_duration.count());
    printf("LTCP descriptor calculation time: %f ms\n", dt_ms);
    printf("Total running time: %ld ms\n", total_duration.count());

    imwrite("result_image.jpg", result_image);

    return 0;
}
