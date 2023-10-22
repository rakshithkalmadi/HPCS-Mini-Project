#include <stdio.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include <chrono>
#include "mpi.h"

using namespace cv;

int b(int a, int b, int c) {
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

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size <= 1) {
        printf("This example requires more than one process. Exiting...\n");
        MPI_Finalize();
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    Mat image;
    if (rank == 0) {
        // Only process 0 loads the image
        image = imread("image1.jpg", IMREAD_GRAYSCALE);

        if (image.empty()) {
            printf("Image not found or could not be opened.\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Bcast(&image.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&image.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = image.rows / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? image.rows : start_row + rows_per_process;

    Mat result_image(image.rows, image.cols, CV_8UC1);

    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < image.cols; x++) {
            int P0 = image.at<uchar>(y, x);
            int P1 = (x < image.cols - 1) ? image.at<uchar>(y, x + 1) : 0;
            int P2 = (x < image.cols - 1 && y < image.rows - 1) ? image.at<uchar>(y + 1, x + 1) : 0;
            int P3 = (y < image.rows - 1) ? image.at<uchar>(y + 1, x) : 0;
            int P4 = (x > 0 && y < image.rows - 1) ? image.at<uchar>(y + 1, x - 1) : 0;
            int P5 = (x > 0) ? image.at<uchar>(y, x - 1) : 0;
            int P6 = (x > 0 && y > 0) ? image.at<uchar>(y - 1, x - 1) : 0;
            int P7 = (y > 0) ? image.at<uchar>(y - 1, x) : 0;
            int C = P0; // Assuming center pixel is P0

            int CP[8];
            CP[0] = b(P7 - C, P0 - C, P1 - C);
            CP[1] = b(P1 - C, P2 - C, P3 - C);
            CP[2] = b(P3 - C, P4 - C, P5 - C);
            CP[3] = b(P5 - C, P6 - C, P7 - C);
            CP[4] = b(P6 - P0, C - P0, P2 - P0);
            CP[5] = b(P4 - P2, C - P2, P0 - P2);
            CP[6] = b(P2 - P4, C - P4, P6 - P4);
            CP[7] = b(P0 - P6, C - P6, P4 - P6);

            // Reverse the CP array and convert it to a decimal value
            int decimal_value = 0;
            for (int i = 7; i >= 0; i--) {
                decimal_value = decimal_value * 2 + CP[i];
            }

            // Set the pixel value in the result image
            result_image.at<uchar>(y, x) = static_cast<uchar>(decimal_value);
        }
    }

    // Gather results from all processes
    MPI_Gather(result_image.data + start_row * image.cols, rows_per_process * image.cols, MPI_UNSIGNED_CHAR,
               result_image.data, rows_per_process * image.cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    auto end_time = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        // Calculate the total running time
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Print the timings
        printf("Total running time: %ld ms\n", total_duration.count());

        // Save the result image
        imwrite("result_image.jpg", result_image);
    }

    MPI_Finalize();

    return 0;
}
