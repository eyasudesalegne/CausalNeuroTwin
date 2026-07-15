#!/usr/bin/env bash
#SBATCH --job-name=cnt-doctor-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=cnt-doctor-cpu-%j.out
#SBATCH --error=cnt-doctor-cpu-%j.err

set -euo pipefail

# Load the verified site environment or execute the approved container here.
causalneurotwin doctor --json
