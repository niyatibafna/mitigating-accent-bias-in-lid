#!/usr/bin/env bash

#$ -N flr_att_ps_train_lid
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/
#$ -m e
#$ -t 1-2
#$ -j y -o qsub_logs/ps_fltrain_attentions_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l h_rt=40:00:00,mem_free=20G,gpu=1,hostname=!r8n04&!r9n08&!r7n04

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

model_key="wav2vec2-xlsr-53-espeak-cv-ft"
dataset_dir="fleurs"
# dataset_dir=None
# per_lang=100
num_epochs=10
batch_size=(128)
evaluate_steps=100
# batch_sizes=(4)
lr=0.0001
num_attention_layers_all=(4 8)
num_attention_layers=${num_attention_layers_all[$SGE_TASK_ID-1]}

lid_model_type="attentions-linear"


output_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/phoneseq_exps/$dataset_dir/$model_key/$lid_model_type-$num_attention_layers/phoneseq_lid_model_outputs/"
mkdir -p $output_dir

logdir="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_phoneseqs/train_logs/$model_key/$lid_model_type"
mkdir -p $logdir
logfile="$logdir/train_lid_attentions-$num_attention_layers-linear.log"

# eval_dataset_dir="edacc"
# eval_dataset_dir="cv"
eval_dataset_dir="fleurs_test"


/home/hltcoe/nbafna/.conda/envs/accent_bias/bin/python lid_with_phoneseqs/train_lid_on_phoneseqs.py \
    --dataset_dir $dataset_dir \
    --num_epochs $num_epochs \
    --batch_size $batch_size \
    --evaluate_steps $evaluate_steps \
    --lr $lr \
    --output_dir $output_dir \
    --lid_model_type $lid_model_type \
    --logfile $logfile \
    --num_attention_layers $num_attention_layers \
    --eval_dataset_dir $eval_dataset_dir \
    --load_trained_from_dir \
    --only_eval 
    # --per_lang $per_lang \
    
    # --load_trained_from_dir

echo "Training LID complete"

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