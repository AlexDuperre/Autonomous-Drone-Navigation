#!/bin/bash

# clear empty files ( trajectories from a point to the same point such as path_0_0_000.h5
find . -name "*.h5" -print0 | xargs -0 -n1  ./clear_empty_trajectories.sh 

# Create new class counter file
> "./class_count.txt"

# Launch data cleaning script on the dataset 
find . -name "*.h5" -print0 | xargs -0 -n1  python3 data_cleaner.py 


