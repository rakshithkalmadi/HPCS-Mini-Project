#include <stdio.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include <chrono>

using namespace cv;
using namespace std;

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

int main(int argc,char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    imwrite("grayscale_image.jpg", image);
    if (image.empty()) {
        printf("Image not found or could not be opened.\n");
        return 1;
    }
	//cout<<image;
	

    Mat result_image(image.rows, image.cols, CV_8UC1); // Create a new image to store the results
	auto image_load_time = std::chrono::high_resolution_clock::now();
	auto image_loading_duration = std::chrono::duration_cast<std::chrono::milliseconds>(image_load_time - start_time);

    for (int y = 1; y < image.rows; y++) {
        for (int x = 1; x < image.cols; x++) 
        {
        	
        			int C = image.at<uchar>(y, x);
            			int P0 = image.at<uchar>(y, x + 1);
            			int P1 = image.at<uchar>(y + 1, x + 1);
            			int P2 = image.at<uchar>(y + 1, x);
            			int P3 = image.at<uchar>(y + 1, x - 1);
            			int P4 = image.at<uchar>(y, x - 1);
            			int P5 = image.at<uchar>(y - 1, x - 1);
            			int P6 = image.at<uchar>(y - 1, x);
            			int P7 = image.at<uchar>(y - 1, x + 1);
            			
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
            			result_image.at<uchar>(y, x)=static_cast<uchar>(decimal_value);
            		}
            	
    }
auto end_time = std::chrono::high_resolution_clock::now();

    auto total_duration = std::chrono::duration<float, std::milli>(end_time - start_time);
    auto image_loading_duration_float = std::chrono::duration<float, std::milli>(image_loading_duration);
    auto ltcp_calculation_duration_float = total_duration - image_loading_duration_float;

    printf("Image loading time: %.3f milliseconds\n", image_loading_duration_float.count());
    printf("LTCP descriptor calculation time: %.3f milliseconds\n", ltcp_calculation_duration_float.count());
    printf("Total running time: %.3f milliseconds\n", total_duration.count());

    imwrite("result_image.jpg", result_image);
	return 0;
}
