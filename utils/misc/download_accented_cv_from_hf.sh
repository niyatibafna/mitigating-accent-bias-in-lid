#!/usr/bin/env bash

#$ -N download_hf
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/misc
#$ -m e
#$ -t 8
#$ -j y -o /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/qsub_logs/download_accented_cv_from_hf_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l h_rt=02:00:00,mem_free=20G

source ~/.bashrc
which python

conda deactivate
conda activate accent_bias
# conda activate /home/hltcoe/rwicks/.conda/envs/sb
which python

echo "HOSTNAME: $(hostname)"
echo
echo CUDA in ENV:
env | grep CUDA

set -x # print out every command that's run with a +
nvidia-smi
export TORCH_DISTRIBUTED_DEBUG=DETAIL

WD="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/misc"
cd $WD

langs=("es" "it" "de" "fr" "hi" "zh" "ar" "ru")
lang=${langs[$SGE_TASK_ID-1]}
echo "Downloading $lang"

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=0,1
python /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/utils/misc/download_accented_cv_from_hf.py $lang

