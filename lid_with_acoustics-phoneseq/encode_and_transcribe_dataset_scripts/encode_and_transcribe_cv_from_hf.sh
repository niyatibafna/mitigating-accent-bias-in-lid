#!/usr/bin/env bash

#$ -N enc_trans_cv
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/
# #$ -pe openmpi 8                   # Request 8 CPU cores
#$ -m e
#$ -t 4
#$ -j y -o qsub_logs/encode_and_transcribe_cv_from_hf/$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l h_rt=10:00:00,mem_free=20G,gpu=1,hostname=!r8n04&!r9n08&!r7n04&!r8n03&!r5n11

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

# dataset_dir="/exp/jvillalba/corpora/voxlingua107"
dataset_dir="cv_from_hf"
# dataset_dir=None
batch_size=16

all_langs=("es" "de" "fr" "it")
lang=${all_langs[$SGE_TASK_ID-1]}

output_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/reps_and_phoneseqs/$dataset_dir/$model_key/"
mkdir -p $output_dir
echo "Encoding and transcribing $lang"

# If output_dir/$lang_transcriptions.jsonl exists, skip
if [ -f "$output_dir/${lang}_transcriptions.jsonl" ]; then
    echo "$lang already transcribed. Skipping."
    exit 0
fi

logdir="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_ssl_units/train_logs/encode_and_transcribe/$dataset_dir/"
mkdir -p $logdir
logfile="$logdir/transcribe_$lang.log"


# mpirun -np $NSLOTS python lid_with_ssl_units/extracting_units_from_training_data.py \
python /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_acoustics-phoneseq/encode_and_transcribe_dataset.py \
    --transcriber_model $transcriber_model \
    --encoder_model $encoder_model \
    --dataset_name $dataset_dir \
    --batch_size $batch_size \
    --output_dir $output_dir \
    --log_file $logfile \
    --lang $lang \