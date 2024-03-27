#include <stdio.h>
#include <opencv2/opencv.h>

int main() {
    // Load an image from a file
    cv::Mat image = cv::imread("your_image.jpg", cv::IMREAD_GRAYSCALE); // Load image as grayscale

    // Check if the image was loaded successfully
    if (image.empty()) {
        printf("Could not open or find the image.\n");
        return -1;
    }

    // Get image dimensions (width and height)
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    // Access grayscale pixel values
    for (int y = 0; y < imageHeight; y++) {
        for (int x = 0; x < imageWidth; x++) {
            // Get the grayscale pixel value at coordinates (x, y)
            uchar pixelValue = image.at<uchar>(y, x);

            // You can now work with the grayscale pixel value here
            printf("Pixel at (%d, %d): Grayscale Value = %d\n", x, y, pixelValue);
        }
    }

    return 0;
}

