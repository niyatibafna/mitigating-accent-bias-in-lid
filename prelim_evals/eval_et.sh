#!/usr/bin/env bash

#$ -N eval_et
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals
#$ -m e
#$ -t 1-107
#$ -j y -o /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals/qsub_logs/eval_et_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l h_rt=5:00:00,mem_free=20G,gpu=1

# Submit to GPU queue
#$ -q gpu.q

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

# module load cuda12.1/toolkit
# module load cudnn/8.4.0.27_cuda11.x
# module load nccl/2.13.4-1_cuda11.7

WD="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals"
cd $WD

all_langs=$(ls /exp/jvillalba/corpora/voxlingua107)
all_langs=($all_langs)
# langs_already_computed=$(ls /exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/$model_key)
# Only extract units for languages that haven't been computed yet
# langs=$(echo $all_langs | tr " " "\n" | grep -v -w -f <(echo $langs_already_computed | tr " " "\n"))
# langs=($langs)
# echo "Number of languages to extract units from:"
# echo ${#langs[@]}
# langs=(haw hi hr ht hu hy ia id it ja ka ko la lb lo lv mi ml mr mt ne no oc ps pt ru sco si sl so sr sv ta tg tk tr uk uz war yo)
lang=${all_langs[$SGE_TASK_ID-1]}
echo "Extracting units for $lang"


export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=0,1
python /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/prelim_evals/eval_et.py $lang

