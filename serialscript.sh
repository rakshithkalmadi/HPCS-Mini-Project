#!/bin/bash

# Compile the program
g++ serial.cpp -O3 -o output `pkg-config --cflags --libs opencv4`

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Set the directory containing image files
IMAGE_DIRECTORY="sampleimages/"

# Set the number of repetitions
REPEATS=50

# Iterate through all image files in the directory
for image_file in $IMAGE_DIRECTORY*.jpg; do
    # Check if the file is a regular file
    if [ -f "$image_file" ]; then
        total_duration=0
        # Print information before running the program
        echo "Running ./output for $image_file $REPEATS times"

        # Run your program and calculate the average duration
        for ((i=1; i<=$REPEATS; i++)); do
            duration=$(./output "$image_file" | grep "LTCP descriptor calculation time" | awk '{print $5}')
           # echo "LTCP descriptor calculation time: $duration ms"

            # Add the duration to the total
            total_duration=$(awk "BEGIN {print $total_duration + $duration}")

            # Print a separator line for better readability
            #echo "------------------------------------------"
        done

        # Calculate the average duration
        average_duration=$(awk "BEGIN {print $total_duration / $REPEATS}")
        echo -e "Average LTCP descriptor calculation time: $average_duration ms\n"
    fi
done


