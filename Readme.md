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

Running Program using MPI (parallel_mpi.cpp):
mpic++ parallel_mpi.cpp -o output_parallel -I /usr/local/include/opencv4 -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs

mpic++ parallel_mpi.cpp -o output_parallel `pkg-config --cflags --libs opencv4`

mpirun -n 5 ./output_parallel

Here -n 5 denotes the number of processes to run this program in parallel, basically the rows of the matrix should be divisible by it. Here for example we have taken 10x10 image which has 10 rows and if we assign 5 processes the each process will be getting 2 rows which will work fine, but if you give the number of process like 4 it will not generate the image properly which leads to black lines in the image.


To increase the number of threads:
mpirun --oversubscribe -np 12 ./output_parallel