#include<iostream>
#include<opencv4/opencv2/core.hpp>
#include<opencv4/opencv2/highgui.hpp>
#include<opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{	//for color image
	//Mat_<Vec3b> image = imread("./image1.jpg", IMREAD_COLOR);
	Mat image = imread("./image1.jpg", IMREAD_GRAYSCALE);
	//for grayscale
	int w = image.cols;
	int h = image.rows;
	
	cout << w << ", " << h<<endl;
	
	//-----------------------------------------
	int threshold(int v) 
	{
    		return (v >= 0) ? 1 : 0;
	}

	// Calculate LTCP descriptor for a single pixel
	int calculateLTCP(const Mat& image, int xc, int yc) 
	{
    		int LTCP = 0;

    		// Define the neighborhood coordinates in clockwise order
    		int coords[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}};

    		for (int i = 0; i < 8; i++) 
    		{
        		int x = xc + coords[i][0];
        		int y = yc + coords[i][1];

        		// Ensure the coordinates are within the image bounds
        		if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) 
        		{
            			// Calculate CPi using the threshold function
            			int CPi = threshold(image.at<uchar>(yc, xc) - image.at<uchar>(y, x));
            			LTCP |= (CPi << i);
        		}
    		}
    		return LTCP;
	}

int main() 
{
	Mat image = imread("your_image.jpg", IMREAD_GRAYSCALE);
	if (image.empty()) 
    	{
        	cout << "Error: Image not found or could not be loaded." << endl;
        	return -1;
    	}

    	// Define the pixel coordinates (xc, yc) for which you want to calculate the LTCP
    	int xc = 50;
    	int yc = 50;

    	// Calculate the LTCP descriptor for the specified pixel
   	int descriptor = calculateLTCP(image, xc, yc);

    	cout << "LTCP Descriptor for pixel (" << xc << ", " << yc << "): " << descriptor << endl;
	//------------------------------------
	return 0;
}
