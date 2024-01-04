#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include<bits/stdc++.h>

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

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    int image[10][10]={ {64,  64,  64,  64,  64,  64,  64,  64,  64,  64},{
  64, 208, 209, 208, 209, 208, 208, 208, 208,  64},{
  64, 209,  64,  64,  64,  64,  64,  64, 209,  64},{
  64, 209,  64, 209, 209, 209, 209,  64, 208,  64},{
  64, 209,  64, 209,  64,  64, 208,  64, 209,  64},{
  64, 209,  64, 208,  64,  64, 209,  64, 208,  64},{
  64, 209,  64, 209, 209, 208, 208,  64, 209,  64},{
  64, 209,  64,  64,  64,  64,  64,  64, 208,  64},{
  64, 209, 208, 209, 209, 208, 208, 209, 209,  64},{
  64,  64,  64,  64,  64,  64,  64,  64,  64,  64}};
    // Mat image = imread("image1.jpg", IMREAD_GRAYSCALE);
    // imwrite("grayscale_image.jpg", image);
    // if (image.empty()) {
    //     printf("Image not found or could not be opened.\n");
    //     return 1;
    // }
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            printf("%d ",image[i][j]);
        }
    }
	auto image_load_time = std::chrono::high_resolution_clock::now();
	auto image_loading_duration = std::chrono::duration_cast<std::chrono::milliseconds>(image_load_time - start_time);

    int result_image[10][10]; // Create a new image to store the results

    for (int y = 1; y < 10; y++) {
        for (int x = 1; x < 10; x++) 
        {
        	
        			int C = image[y][x];
            			int P0 = image[y][x + 1];
            			int P1 = image[y + 1][x + 1];
            			int P2 = image[y + 1][x];
            			int P3 = image[y + 1][x - 1];
            			int P4 = image[y][x - 1];
            			int P5 = image[y - 1][x - 1];
            			int P6 = image[y - 1][x];
            			int P7 = image[y - 1][x + 1];
            			
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
            			result_image[y][x]=static_cast<int>(decimal_value);
            		}
            	
    }
	auto end_time = std::chrono::high_resolution_clock::now();

    	// Calculate the total running time
	auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    	// Print the timings
	printf("Image loading time: %ld ms\n", image_loading_duration.count());
	printf("LTCP descriptor calculation time: %ld ms\n", total_duration.count() - image_loading_duration.count());
	printf("Total running time: %ld ms\n", total_duration.count());
    	// Save the result image
	// imwrite("result_image.jpg", result_image);
    for(int i=1;i<10;i++){
        for(int j=1;j<10;j++){
            printf("%d \t",result_image[i][j]);
        }
        std::cout<<"\n";
    }

    return 0;
}