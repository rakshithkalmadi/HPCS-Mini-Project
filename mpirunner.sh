#!/bin/bash

# Set the path to your program
PROGRAM="./output_parallel"

# Set the directory containing image files
IMAGE_DIRECTORY="new/"

# Set the range of processors to loop through
for NUM_PROCESSES in {2..12..2}; do
    # Iterate through all image files in the directory
    for image_file in $IMAGE_DIRECTORY*.jpg; do
        # Check if the file is a regular file
        if [ -f "$image_file" ]; then
            # Print information before running the program
            echo "Running $PROGRAM for $image_file with $NUM_PROCESSES processes"

            # Run your program with mpirun
            mpirun -n $NUM_PROCESSES $PROGRAM "$image_file"

            # Print a separator line for better readability
            echo "------------------------------------------"
        fi
    done
done
