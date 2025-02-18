#!/usr/bin/env bash

#$ -N eval-sslembed_train_lid
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/
#$ -m e
#$ -t 1-3
#$ -j y -o qsub_logs/eval-sslembed_train_attentions_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l h_rt=2:00:00,mem_free=20G,gpu=1,hostname=!r8n04&!r9n08&!r7n04

# Submit to GPU queue
#$ -q gpu.q

source ~/.bashrc
which python

conda deactivate
conda activate accent_bias
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

WD="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/"
cd $WD

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=0,1


encoder_model="ecapa-tdnn"
transcriber_model="facebook/wav2vec2-xlsr-53-espeak-cv-ft"
ssl_model="facebook/wav2vec2-base"
model_key="ecapa-tdnn_wav2vec2-xlsr-53-espeak-cv-ft_wav2vec2-base-layer8-1000"

kmeans_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/wav2vec2-base-layer8-1000/global_kmeans/"

dataset_dir="vl107"
# Here, dataset_dir refers to the dataset with which the kmeans clustering was done. 
# The eval units will therefore also be in the vl107 directory, under $model_key/cv_dataset/ for example
save_dataset_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/reps_phoneseqs_duseqs_exps/vl107/$model_key/${dataset_dir}_dataset/"
mkdir -p $save_dataset_dir

# dataset_dir=None
# per_lang=100
num_epochs=25
batch_size=(128)
evaluate_steps=100
# batch_sizes=(4)
lr=0.0001
num_attention_layers_all=(4 8)
# num_attention_layers=${num_attention_layers_all[$SGE_TASK_ID-1]}
num_attention_layers=8

lid_model_type="attentions-linear"


output_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/reps_phoneseqs_duseqembed_exps/$dataset_dir/$model_key/$lid_model_type-$num_attention_layers/reps-phoneseq-duseqs_nocnn_lid_model_outputs/"
mkdir -p $output_dir

logdir="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_reps-phoneseqs/train_logs/$model_key/$lid_model_type"
mkdir -p $logdir
logfile="$logdir/train_lid_attentions-$num_attention_layers-linear.log"

eval_dataset_dirs=("cv" "fleurs_test" "edacc")
eval_dataset_dir=${eval_dataset_dirs[$SGE_TASK_ID-1]}
# eval_dataset_dir="edacc"
# eval_dataset_dir="cv"
# eval_dataset_dir="fleurs_test"
save_eval_dataset_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/reps_phoneseqs_duseqs_exps/$dataset_dir/$model_key/${eval_dataset_dir}_dataset/"
mkdir -p $save_eval_dataset_dir


/home/hltcoe/nbafna/.conda/envs/accent_bias/bin/python /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_reps-sslunits/train_lid_with_sslembed_et.py \
    --dataset_name $dataset_dir \
    --save_dataset_dir $save_dataset_dir \
    --transcriber_model $transcriber_model \
    --encoder_model $encoder_model \
    --ssl_model $ssl_model \
    --kmeans_dir $kmeans_dir \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --evaluate_steps $evaluate_steps \
    --lr $lr \
    --output_dir $output_dir \
    --lid_model_type $lid_model_type \
    --logfile $logfile \
    --num_attention_layers $num_attention_layers \
    --eval_dataset_name $eval_dataset_dir \
    --save_eval_dataset_dir $save_eval_dataset_dir \
    --load_trained_from_dir \
    --only_eval

echo "Training LID complete"

