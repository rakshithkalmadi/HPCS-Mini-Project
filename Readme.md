To install opencv:
sudo apt update
sudo apt install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig


To run the serial program in pc:
g++ serial.cpp -o output -I /usr/local/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs

Format: g++ your_program.cpp -o your_program -I /path/to/opencv/include -L /path/to/opencv/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs

