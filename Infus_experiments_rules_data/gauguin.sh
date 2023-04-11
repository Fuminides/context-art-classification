#!/bin/bash
#$ -q cal.q
#$ -cwd

source activate fuzzyexp2
python ./gogh_gauguin.py 1000 30 20 3 2
