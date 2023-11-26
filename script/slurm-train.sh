#!/bin/bash
#SBATCH -p dgx2q
#SBATCH  --job-name=sec
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time 00-02:00:00    # time (D-HH:MM:SS)
#SBATCH -o ./output/%j.out # STDOUT
# SBATCH -e ./output/%j.err # STDERR

ulimit -s 10240
echo "Job started at:" `date +"%Y-%m-%d %H:%M:%S"`

# module purge
# module load slurm/20.02.7
# module load python37
# # module load tensorflow2-py37-cuda10.2-gcc8/2.5.0  
# module load ml-pythondeps-py37-cuda11.2-gcc8/4.7.8
# module load pytorch-extra-py37-cuda11.2-gcc8/1.9.1
# module load python-mpi4py-3.0.3
# module listls


module purge
module load slurm/21.08.8
module load python39
# module load pytorch-py37-cuda10.2-gcc8/1.8.1  # this module is no more available in the module list.
module load tensorflow2-py39-cuda11.8-gcc11/2.11.0
# module load pytorch-py37-cuda11.2-gcc8/1.9.1
#  module load pytorch-py37-cuda10.2-gcc8/1.8.1
module list



source venv/bin/activate
which python3
python3 --version

srun script/train.sh

echo "Job ended at:" `date +"%Y-%m-%d %H:%M:%S"`