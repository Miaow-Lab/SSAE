#!/bin/bash
module load miniforge3/25.11.0-1 cuda/12.4
source activate pytorch 

echo "Start sampling..."

bash scripts/run_n2g.sh
