#!/bin/sh
#
#SBATCH --job-name="hydroelasticity"
#SBATCH --partition=genoa
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o stdout/slurm-%j-%4t.out
#SBATCH -e stdout/slurm-%j-%4t.err

source ../compile/modules.sh
TEST_CASE="${TEST_CASE:-$1}"
if [ -z "$TEST_CASE" ]; then
	echo "TEST_CASE is not set"
	exit 1
fi
export TEST_CASE
echo "Starting case: $TEST_CASE"
julia --project=.. --check-bounds=no job.jl