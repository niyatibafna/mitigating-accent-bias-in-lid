#!/usr/bin/env bash

#$ -N vl_extract_training_units_alllangs
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/
# #$ -pe openmpi 8                   # Request 8 CPU cores
#$ -m e
#$ -t 6
#$ -j y -o qsub_logs/xlsr_extracting_units_alllangs_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l h_rt=60:00:00,mem_free=20G,gpu=1,hostname=!r8n04&!r9n08&!r7n04&!r8n03

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

# model_name="facebook/wav2vec2-base"
layer=21
model_name="facebook/wav2vec2-large-xlsr-53"
model_key="facebook/wav2vec2-large-xlsr-53-layer$layer"
# model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus"
# model_key="wavlm-base-layer8"

# layer=8

# dataset_dir="/exp/jvillalba/corpora/voxlingua107"
dataset_dir="vl107"
# dataset_dir=None
per_lang=500
batch_size=16
n_clusters_all=(100 250 500 750 1000 5000 10000 20000 30000)
n_clusters=${n_clusters_all[$SGE_TASK_ID-1]}


kmeans_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/$model_key-$n_clusters/global_kmeans/"
# output_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/$model_key/$lang"
echo "Extracting units for $lang"

logdir="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_ssl_units/train_logs"
mkdir -p $logdir
logfile="$logdir/extracting_units_from_training_data_$lang_$n_clusters.log"


# mpirun -np $NSLOTS python lid_with_ssl_units/extracting_units_from_training_data.py \
python lid_with_ssl_units/extracting_units_from_training_data.py \
    --model_name $model_name \
    --layer $layer \
    --dataset_dir $dataset_dir \
    --kmeans_dir $kmeans_dir \
    --batch_size $batch_size \
    --output_dir $kmeans_dir \
    --n_clusters $n_clusters \
    --log_file $logfile \
    --per_lang $per_lang \


# parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="Model name")
#     parser.add_argument("--layer", type=int, default=8, help="Layer to extract representations from")
#     parser.add_argument("--dataset_dir", type=str, default="/exp/jvillalba/corpora/voxlingua107", help="Directory containing audio files")
#     parser.add_argument("--per_lang", type=int, default=None, help="Number of audio files per language")
#     parser.add_argument("--lang", type=str, default=None, help="Language to extract audio files from")
#     parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
#     parser.add_argument("--output_dir", type=str, default="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs", help="Output directory")