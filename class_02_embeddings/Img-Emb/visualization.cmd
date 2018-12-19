#!/bin/bash
# @ job_name= visualization
# @ initialdir= .
# @ output= logs/visualization.out
# @ error= logs/visualization.err
# @ total_tasks= 1
# @ gpus_per_node = 1
# @ cpus_per_task= 1
# @ features = k80
# @ wall_clock_limit = 00:05:00

::Code for running in MT with Python 2.7.12
module purge
module load merovingian

merovingian2712 visualization.py
