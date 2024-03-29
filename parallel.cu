#include <stdio.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include <chrono>

using namespace cv;
using namespace std;

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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < cols-1 && y > 0 && y < rows-1) {
        int C = input[y * cols + x];
        int P0 = input[y * cols + x + 1];
        int P1 = input[(y + 1) * cols + x + 1];
        int P2 = input[(y + 1) * cols + x];
        int P3 = input[(y + 1) * cols + x - 1];
        int P4 = input[y * cols + x - 1];
        int P5 = input[(y - 1) * cols + x - 1];
        int P6 = input[(y - 1) * cols + x];
        int P7 = input[(y - 1) * cols + x + 1];

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

int main(int argc, char* argv[]) {
  if (argc != 2) {
        printf("Usage: %s <input_image>\n", argv[0]);
        return 1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();

    //create events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Image not found or could not be opened.\n");
        return 1;
    }
    //cout<<image;
    auto image_load_time = std::chrono::high_resolution_clock::now();
    auto image_loading_duration = std::chrono::duration_cast<std::chrono::milliseconds>(image_load_time - start_time);

    int rows = image.rows;
    int cols = image.cols;

    Mat result_image(rows, cols, CV_8UC1);

    uchar* d_input, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(uchar));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(uchar));

    cudaMemcpy(d_input, image.data, rows * cols * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    //record events around kernel launch
    cudaEventRecord(event1, 0);
    ltpcKernel<<<gridDim, blockDim>>>(d_input, d_output, rows, cols);
    cudaEventRecord(event2, 0);

    //synchronize
    cudaEventSynchronize(event1); //optional
    cudaEventSynchronize(event2); //wait for the event to be executed!

    //calculate time
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
    //cout<<result_image;

    return 0;
}
