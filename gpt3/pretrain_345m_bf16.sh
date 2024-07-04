#!/bin/bash -l
#SBATCH --job-name="pretrain_bf16_345m"
#SBATCH --nodes=4                  # number of nodes
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --gpus-per-node=4          # number of gpus per node
#SBATCH --partition=normal
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00            # total run time limit (HH:MM:SS)
# Runs the "345M" parameter model

set -e

GLOBAL_VARS="\
# Setting the environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_MR_CACHE_MONITOR="userfaultfd"

# Extra debugging flags, slow down training
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
"

FP_TYPE=bf16

# Distributed training variables
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=4
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

BASE_PATH="/capstor/scratch/cscs/ctianche/playground/low-precision-testbed/msamp_reprod/msamp_examples/gpt3"
CHECKPOINT_PATH=${BASE_PATH}/checkpoints/gpt_345m_bf16
LOG_PATH=${BASE_PATH}/logs/gpt_345m_bf16.log
TENSORBOARD_PATH=${BASE_PATH}/tensorboard/gpt_345m_bf16

VOCAB_FILE=${BASE_PATH}/data/gpt2-vocab.json
MERGE_FILE=${BASE_PATH}/data/gpt2-merges.txt
DATA_PATH=${BASE_PATH}/data/wikipedia_text_document

DISTRIBUTED_ARGS=" \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank \${NODE_RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="\
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 16 \
    --global-batch-size 256 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --no-gradient-accumulation-fusion \
    --use-distributed-optimizer \
"

DATA_ARGS="\
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
"

OUTPUT_ARGS="\
    --log-interval 100 \
    --save-interval 50000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

TB_ARGS="\
    --tensorboard-dir ${TENSORBOARD_PATH} \
    --log-memory-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    "

RUN="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/../third_party/Megatron-LM/pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${TB_ARGS} \
    --transformer-impl transformer_engine \
    --save ${CHECKPOINT_PATH} \
    --load ${CHECKPOINT_PATH} \
    "

srun --mpi=pmi2 --environment=msamp bash -c "
export NODE_RANK=\${SLURM_NODEID}
${GLOBAL_VARS}
cd ${BASE_PATH}
echo ${RUN}
${RUN}
"
