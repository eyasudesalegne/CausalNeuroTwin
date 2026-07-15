#!/usr/bin/env bash
#SBATCH --job-name=cnt-doctor-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=cnt-doctor-gpu-%j.out
#SBATCH --error=cnt-doctor-gpu-%j.err

set -euo pipefail

# The GPU resource syntax and environment must be verified on the target system.
causalneurotwin doctor --json
