#!/bin/bash
#SBATCH --job-name=quant
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=23:55:00
#SBATCH --mail-user=sl2998@princeton.edu
#SBATCH --partition=pli-c
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

set -x  # Exit immediately if any command fails

run_experiment() {
    local PRUNER=rand
    local MODEL=vgg16
    local DATASET=cifar10
    local MODE=singleshot
    local COMPRESSION=$1
    local MODEL_CLASS=lottery
    local PRE_EPOCHS=0
    local POST_EPOCHS=100
    local NAME="${PRUNER}-${MODEL}-${DATASET}-${MODE}-${MODEL_CLASS}-c${COMPRESSION}-pre${PRE_EPOCHS}-post${POST_EPOCHS}-amp"
    
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
        --quantization \
        --expid "${NAME}" 2>&1 | tee logs/${NAME}.log
}
# Main execution
declare -a COMPRESSION=(1 0.5 0.2 0.1 0.05 2)

for compression in "${COMPRESSION[@]}"; do
    run_experiment \
        "${compression}"
done

echo "All experiments completed successfully"

set +x