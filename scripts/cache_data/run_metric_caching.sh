export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="xxx/dataset/maps"
export NAVSIM_EXP_ROOT="xxx/exp"
export NAVSIM_DEVKIT_ROOT="xxx"
export OPENSCENE_DATA_ROOT="xxx/dataset"
export HYDRA_FULL_ERROR=1
export HF_HOME=xxx/hf_cache

TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=xxx/dataset/cache/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH
