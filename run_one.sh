#!/bin/bash

# In the shell, or in the script calling this script, set these environment variables.
echo $FLASHIDNUM $PATH 



/home/kebrunne/miniconda3/envs/PAWS/bin/python knb_mc_model_3dCOMMAS337.py --flashid=$FLASHIDNUM --microphysics_file=$PATH



