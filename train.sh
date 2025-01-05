#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 2:00:00
#SBATCH -o ./ship-%j.out

python run.py --config configs/nerf/ship.py --render_test
