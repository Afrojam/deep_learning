#!/bin/bash
# @ job_name= image_embeddings
# @ initialdir= .
# @ output= logs/image_embeddings.out
# @ error= logs/image_embeddings.err
# @ total_tasks= 1
# @ gpus_per_node = 1
# @ cpus_per_task= 1
# @ features = k80
# @ wall_clock_limit = 00:10:00

::Code for running in MT with Python 2.7.12
module purge
module load merovingian

merovingian2712 image_embeddings.py
