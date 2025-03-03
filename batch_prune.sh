#!/bin/bash
#SBATCH --job-name=prune
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --mail-user=sl2998@princeton.edu
#SBATCH --partition=pli-c
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

set -x  # Exit immediately if any command fails

module purge
module load anaconda3/2024.2
conda activate hw

run_experiment() {
    local PRUNER=$1
    local MODEL=$2
    local DATASET=$3
    local MODE=$4
    local COMPRESSION=$5
    local MODEL_CLASS=$6
    local PRE_EPOCHS=$7
    local POST_EPOCHS=$8
    local NAME="${PRUNER}-${MODEL}-${DATASET}-${MODE}-${MODEL_CLASS}-c${COMPRESSION}-pre${PRE_EPOCHS}-post${POST_EPOCHS}"
    
    echo "Running experiment: ${NAME}"
    python main.py \
        --model-class "${MODEL_CLASS}" \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --experiment "${MODE}" \
        --pruner "${PRUNER}" \
        --compression "${COMPRESSION}" \
        --pre-epochs "${PRE_EPOCHS}" \
        --post-epochs "${POST_EPOCHS}" \
        --expid "${NAME}" 2>&1 | tee logs/${NAME}.log
}

# Main execution
declare -a MODEL_CLASSES=("default" "lottery")

declare -a COMPRESSION=(1 0.5 0.2 0.1 0.05 2)

# Experiment configurations (pruner, model, dataset, mode, compression)
declare -a EXPERIMENT_CONFIGS=(
    "rand vgg16 cifar10 singleshot lottery 0 100"
    "mag vgg16 cifar10 singleshot lottery 200 100"
    "snip vgg16 cifar10 singleshot lottery 0 100"
    "grasp vgg16 cifar10 singleshot lottery 0 100"
    "synflow vgg16 cifar10 singleshot lottery 0 100"
    "rand fc mnist singleshot default 0 10"
    "mag fc mnist singleshot default 200 10"
    "snip fc mnist singleshot default 0 10"
    "grasp fc mnist singleshot default 0 10"
    "synflow fc mnist singleshot default 0 10"
)

for config in "${EXPERIMENT_CONFIGS[@]}"; do
    # Split configuration string into components
    read -r pruner model dataset mode model_class pre_epochs post_epochs <<<"${config}"
    for compression in "${COMPRESSION[@]}"; do
        run_experiment \
            "${pruner}" \
            "${model}" \
            "${dataset}" \
            "${mode}" \
            "${compression}" \
            "${model_class}" \
            "${pre_epochs}" \
            "${post_epochs}"
    done
done

echo "All experiments completed successfully"

set +x