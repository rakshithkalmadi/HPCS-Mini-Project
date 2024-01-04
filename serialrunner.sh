#!/bin/bash

# Set the path to your program
PROGRAM="./output"

# Set the directory containing image files
IMAGE_DIRECTORY="new/"

# Iterate through all image files in the directory
for image_file in $IMAGE_DIRECTORY*.jpg; do
    # Check if the file is a regular file
    if [ -f "$image_file" ]; then
        # Print information before running the program
        echo "Running $PROGRAM for $image_file serially"

        # Run your program with mpirun
        $PROGRAM "$image_file"

        # Print a separator line for better readability
        echo "------------------------------------------"
    fi
done
