#!/usr/bin/env bash

#$ -N per_lang20000_extract_training_units
#$ -wd /home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/
#$ -m e
#$ -t 1-107
#$ -j y -o qsub_logs/extracting_units_from_training_data_20000/$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l h_rt=10:00:00,mem_free=20G,gpu=1,hostname=!r8n04&!r9n08&!r7n04
#$ -hold_jid 12861472

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
model_key="wav2vec2-base-layer8"
# model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus"
# model_key="wavlm-base-layer8"
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

layer=8
# dataset_dir="/exp/jvillalba/corpora/voxlingua107"
dataset_dir="vl107"
# dataset_dir=None
# per_lang=5000
batch_size=8
n_clusters=20000

kmeans_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/$model_key-$n_clusters/global_kmeans/"
output_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/vl107/$model_key-$n_clusters/training_units/$lang"
# output_dir="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs/$model_key/$lang"
echo "Extracting units for $lang"

logdir="/home/hltcoe/nbafna/projects/mitigating-accent-bias-in-lid/lid_with_ssl_units/train_logs"
mkdir -p $logdir
logfile="$logdir/extracting_units_from_training_data_$lang.log"

# python lid_with_ssl_units/extracting_units_from_training_data.py \
python lid_with_ssl_units/extracting_units_from_training_data.py \
    --model_name $model_name \
    --layer $layer \
    --dataset_dir $dataset_dir \
    --kmeans_dir $kmeans_dir \
    --lang $lang \
    --batch_size $batch_size \
    --output_dir $output_dir \
    --n_clusters $n_clusters \
    --log_file $logfile \
    # --per_lang $per_lang \

# parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base", help="Model name")
#     parser.add_argument("--layer", type=int, default=8, help="Layer to extract representations from")
#     parser.add_argument("--dataset_dir", type=str, default="/exp/jvillalba/corpora/voxlingua107", help="Directory containing audio files")
#     parser.add_argument("--per_lang", type=int, default=None, help="Number of audio files per language")
#     parser.add_argument("--lang", type=str, default=None, help="Language to extract audio files from")
#     parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
#     parser.add_argument("--output_dir", type=str, default="/exp/nbafna/projects/mitigating-accent-bias-in-lid/wav2vec2_intermediate_outputs", help="Output directory75