export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="xxx/dataset/maps"
export NAVSIM_EXP_ROOT="xxx/exp"
export NAVSIM_DEVKIT_ROOT="xxx"
export OPENSCENE_DATA_ROOT="xxx/dataset"
export HYDRA_FULL_ERROR=1
export HF_HOME=xxx/hf_cache


train_test_split=navtest   #warmup_two_stage   # navmini

synthetic_sensor_path=$OPENSCENE_DATA_ROOT/${train_test_split}/sensor_blobs
synthetic_scenes_path=$OPENSCENE_DATA_ROOT/${train_test_split}/synthetic_scene_pickles

experiment_name=feature_caching_${train_test_split}
save_dir="xxx/dataset/cache/feature_cache/${train_test_split}"
experiment_uid=$(date +%Y.%m.%d.%H.%M.%S)
output_dir="${NAVSIM_EXP_ROOT}/exp_logs/${experiment_name}/${experiment_uid}"
num_workers=8
chunksize=1



echo "======== start caching feature data ========"
echo "output path: $save_dir"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_negwam.py \
    train_test_split=$train_test_split \
    output_dir=$output_dir \
    save_dir=$save_dir \
    experiment_name=$experiment_name \
    num_workers=$num_workers \
    chunksize=$chunksize \
    synthetic_sensor_path=$synthetic_sensor_path \
    synthetic_scenes_path=$synthetic_scenes_path