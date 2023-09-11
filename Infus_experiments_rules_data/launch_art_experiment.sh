#!/bin/bash
#$ -q cal.q
#$ -cwd
#$ -t 1-20

source activate fuzzyexp2
python ./art_demo.py 1000 30 20 3 2
