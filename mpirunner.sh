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

            # Initialize total duration for average calculation
            total_duration=0

            # Run your program with mpirun 5 times
            for ((i=1; i<=5; i++)); do
                echo "Run $i:"
                # Capture the LTCP descriptor calculation time
                duration=$(mpirun -n $NUM_PROCESSES $PROGRAM "$image_file" | grep "LTCP descriptor calculation time" | awk '{print $5}')
                echo "LTCP descriptor calculation time: $duration ms"
                
                # Add the duration to the total
                total_duration=$((total_duration + duration))

                # Print a separator line for better readability
                echo "------------------------------------------"
            done

            # Calculate the average duration
            average_duration=$((total_duration / 5))
            echo -e "Average LTCP descriptor calculation time: $average_duration ms\n"
        fi
    done
done
