#!/usr/bin/env bash

#$ -N cv_eval_att8_lid
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/
#$ -m e
#$ -t 1-3
#$ -j y -o qsub_logs/eval_attentions8_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l h_rt=40:00:00,mem_free=20G,gpu=1,hostname=!r8n04&!r9n08&!r7n04
#$ -hold_jid 12868143

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

model_name="facebook/wav2vec2-base"
units_all=(500 1000 10000)
units=${units_all[$SGE_TASK_ID-1]}

model_key="wav2vec2-base-layer8-$units"
layer=8
# model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus"
# model_key="wavlm-base-layer8"

# dataset_dir="/exp/jvillalba/corpora/voxlingua107"
dataset_dir="vl107"
# dataset_dir=None
# per_lang=None
num_epochs=200
batch_size=(128)
# batch_sizes=(4)
lr=0.0001
num_attention_layers=8

lid_model_type="cnn-attentions-linear"


kmeans_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/$model_key/global_kmeans/"
training_units_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/$model_key/training_units/"
output_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/$model_key/$lid_model_type-$num_attention_layers/lid_model_outputs/"

logdir="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_ssl_units/train_logs/$model_key/$lid_model_type"
mkdir -p $logdir
logfile="$logdir/eval_lid_cnn-attentions-$num_attention_layers-linear.log"

# CHANGE EVAL_UNITS_DIR TOO IF CHANGING THIS: eval_units_dir: eval_units for edacc, cv_eval_units for cv
# eval_dataset_dir="edacc" 
eval_dataset_dir="cv"
eval_units_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/$model_key/cv_eval_units/"

/home/hltcoe/nbafna/.conda/envs/accent_bias/bin/python lid_with_ssl_units/train_lid.py \
    --model_name $model_name \
    --layer $layer \
    --dataset_dir $dataset_dir \
    --training_units_dir $training_units_dir \
    --kmeans_dir $kmeans_dir \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --lr $lr \
    --output_dir $output_dir \
    --lid_model_type $lid_model_type \
    --logfile $logfile \
    --num_attention_layers $num_attention_layers \
    --eval_dataset_dir $eval_dataset_dir \
    --eval_units_dir $eval_units_dir \
    --only_eval \
    --load_trained_from_dir \
    # --per_lang $per_lang \

echo "Evaluating LID complete"

# parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="Model name")
#     parser.add_argument("--layer", type=int, default=8, help="Layer to extract representations from")
#     parser.add_argument("--dataset_dir", type=str, default="/exp/jvillalba/corpora/voxlingua107", help="Directory containing audio files")
#     parser.add_argument("--per_lang", type=int, default=None, help="Number of audio files per language")
#     parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
#     parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
#     parser.add_argument("--output_dir", type=str, default="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs", help="Output directory")
#     parser.add_argument("--load_trained_from_dir", action="store_true", help="Load the model from output_dir")
#     parser.add_argument("--lid_model_type", type=str, default="linear", help="Type of model to train")
#     return parser.parse_args()