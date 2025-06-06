#!/usr/bin/env bash

#$ -N eval_reps-ps_train_lid
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/
#$ -m e
#$ -t 1-2
#$ -j y -o qsub_logs/eval_reps-ps_train_attentions_$TASK_ID.out

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
model_key="ecapa-tdnn_wav2vec2-xlsr-53-espeak-cv-ft"

dataset_dir="vl107"
save_dataset_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/reps_and_phoneseqs/$dataset_dir/$model_key/"
mkdir -p $save_dataset_dir

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


output_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/reps-phoneseq_exps/$dataset_dir/$model_key/$lid_model_type-$num_attention_layers/reps-phoneseq_lid_model_outputs/"
mkdir -p $output_dir

logdir="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_reps-phoneseqs/train_logs/$model_key/$lid_model_type"
mkdir -p $logdir
logfile="$logdir/train_lid_attentions-$num_attention_layers-linear.log"

# eval_dataset_dir="edacc"
# eval_dataset_dir="cv"
# eval_dataset_dir="fleurs_test"
# eval_dataset_dir="nistlre"
eval_dataset_dir="cv_from_hf"

save_eval_dataset_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/reps_and_phoneseqs/$eval_dataset_dir/$model_key/"
mkdir -p $save_eval_dataset_dir


/home/hltcoe/nbafna/.conda/envs/accent_bias/bin/python /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_acoustics-phoneseq/train_lid_with_acoustics_phoneseq.py \
    --dataset_name $dataset_dir \
    --save_dataset_dir $save_dataset_dir \
    --transcriber_model $transcriber_model \
    --encoder_model $encoder_model \
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

# parser.add_argument("--dataset_name", type=str, required=True, help="Directory containing transcribed audio files")
    
#     parser.add_argument("--transcriber_model", type=str, required=True, help="Model used to transcribe the audio files")
#     parser.add_argument("--encoder_model", type=str, required=True, help="Model used to encode the audio files for acoustic representations")
#     parser.add_argument("--save_dataset_dir", type=str, required=True, help="Directory to save the dataset")

#     parser.add_argument("--per_lang", type=int, default=None, help="Number of audio files per language")
#     parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
#     parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
#     parser.add_argument("--evaluate_steps", type=int, default=None, help="Evaluate every n steps")
#     parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
#     parser.add_argument("--output_dir", type=str, default="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs", help="Output directory for LID model")
#     parser.add_argument("--load_trained_from_dir", action="store_true", help="Load the model from output_dir")
#     parser.add_argument("--lid_model_type", type=str, default="linear", help="Type of model to train")
#     parser.add_argument("--logfile", type=str, default="train.log", help="Log file")
#     parser.add_argument("--num_attention_layers", type=int, default=None, help="Number of attention layers")

#     parser.add_argument("--only_eval", action="store_true", help="Only evaluate the model")
#     parser.add_argument("--eval_dataset_name", type=str, default=None, help="Directory containing evaluation dataset")
#     parser.add_argument("--save_eval_dataset_dir", type=str, default=None, help="Directory to save the evaluation dataset")