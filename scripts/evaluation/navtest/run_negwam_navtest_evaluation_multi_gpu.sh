# ======= Environment Variables for navsim =======
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="xxx/dataset/maps"
export NAVSIM_EXP_ROOT="xxx/exp"
export NAVSIM_DEVKIT_ROOT="xxx/navsim"
export OPENSCENE_DATA_ROOT="xxx/dataset"
export HYDRA_FULL_ERROR=1
export HF_HOME=xxx/hf_cache


# ======= NCCL Configuration =======
# for 5090 cluster
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000
export NCCL_IB_GID_INDEX=5
export NCCL_IN_TC=106


# ======= Evaluation Configuration =======
split=navtest
metric_cache_path=$OPENSCENE_DATA_ROOT/cache/metric_cache/${split}
dataset_path=$OPENSCENE_DATA_ROOT/cache/feature_cache/${split}
agent=negwam_agent
checkpoint_path=xxx/exp/negwam_agent/negwam_agent_training_navtrain/2026.07.23.14.17.17/rank_0/checkpoint_step_10000/transformer
experiment_name=${agent}_eval_${split}
num_video_steps=3
num_action_steps=10

experiment_uid=$(date +%Y.%m.%d.%H.%M.%S)
output_dir="${NAVSIM_EXP_ROOT}/${agent}/${experiment_name}/ablation_steps/${experiment_uid}/rank_${RANK}"


echo "Starting evaluation script..."

# ======== distributed evaluation configuration =======
GPUS_PER_NODE=8
NUM_NODES=3
NODE_RANK=${NODE_RANK:-${RANK:-0}}
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export MASTER_ADDR
export MASTER_PORT
export NODE_RANK
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
export RANK=$((NODE_RANK * GPUS_PER_NODE))

echo "========================================"
echo "Distributed Configuration"
echo "========================================"
echo "NUM_NODES:        ${NUM_NODES}"
echo "NODE_RANK:        ${NODE_RANK}"
echo "GPUS_PER_NODE:    ${GPUS_PER_NODE}"
echo "MASTER_ADDR:      ${MASTER_ADDR}"
echo "MASTER_PORT:      ${MASTER_PORT}"
echo "WORLD_SIZE:       $((NUM_NODES * GPUS_PER_NODE))"
echo "========================================"


torchrun --nnodes=${NUM_NODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage_negwam_multi_gpu.py \
    train_test_split=$split \
    gpu=true \
    agent=$agent \
    experiment_name=$experiment_name \
    metric_cache_path=$metric_cache_path \
    output_dir=$output_dir \
    worker=single_machine_thread_pool \
    agent.config.dataset_path=$dataset_path \
    agent.config.checkpoint_path=$checkpoint_path \
    agent.config.num_video_steps=$num_video_steps \
    agent.config.num_action_steps=$num_action_steps
