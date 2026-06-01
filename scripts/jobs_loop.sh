#!/bin/bash

INITIAL_CASE=1
FINAL_CASE=10
rm -rf stdout
mkdir -p stdout
for i in $(seq $INITIAL_CASE $FINAL_CASE)
do
  echo "case: $i"
  sbatch --export=ALL,FINAL_CASE=$FINAL_CASE,TEST_CASE=$i job.sh
done