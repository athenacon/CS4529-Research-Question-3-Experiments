#!/bin/bash --login
#SBATCH --nodes=1 # number of nodes
#SBATCH --cpus-per-task=6 # number of cores
#SBATCH --mem=20G # memory pool for all cores
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --time=150:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=u16ak20@abdn.ac.uk
#SBATCH --signal=SIGUSR1@90

nvidia-smi

module load miniconda3
conda activate testenv

which python
python --version

srun /home/u16ak20/.conda/envs/testenv/bin/python main_vicreg.py --exp-dir "pre-training/" --arch resnet18 --epochs 1000 --batch-size 128 --base-lr 0.2 --data-dir "smallNorb"

