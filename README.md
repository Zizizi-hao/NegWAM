

# Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified World-Action Model

<p align="left">
  <a href="https://arxiv.org/abs/2611.08111"><img src="https://img.shields.io/badge/arXiv-2611.08111-red?style=flat&amp;logo=arxiv" alt="Paper on arXiv"></a>
  <a href="https://huggingface.co/Zizizi-hao/NegWAM"><img src="https://img.shields.io/badge/HuggingFace-Model-orange?style=flat&amp;logo=huggingface" alt="Model &amp; Data on Hugging Face"></a>
</p>

<p>
  <strong>Zihao Zhang</strong><sup>1,2</sup> ·
  <strong>Xinyu Zhang</strong><sup>1</sup> ·
  <strong>Bing Zeng</strong><sup>1</sup> ·
  <strong>Xingyu Chen</strong><sup>3</sup> ·
  <strong>Zhenning Li</strong><sup>4</sup> ·
  <strong>Bin Yu</strong><sup>5</sup> ·
  <strong>Ziyuan Pu</strong><sup>1</sup>
</p>
<p>
  <sup>1</sup> Southeast University &nbsp;·&nbsp; <sup>2</sup> Shenzhen Loop Area Institute<br>
  <sup>3</sup> Guangxi University &nbsp;·&nbsp; <sup>4</sup> University of Macau &nbsp;·&nbsp; <sup>5</sup> Beihang University
</p>



---

<p align="center">
  <img src="https://ik.imagekit.io/zizizihao/NegWAM/figure_02_methods.jpg" width="95%">
</p>

World-action models (WAMs) jointly model future traffic states and driving actions for end-to-end autonomous driving in intelligent transportation systems. However, their action learning remains dominated by expert-trajectory imitation, leaving unsafe alternatives close to expert trajectories insufficiently constrained. To improve WAM robustness in safety-critical scenarios, we propose **NegWAM**, a safety post-training framework for unified WAMs that learns explicit safety boundaries from *near-expert failure trajectories*. NegWAM constructs negative trajectories that are geometrically close to the expert trajectory yet carry safety risks, uses them as repulsive anchors to steer predictions away from dangerous behavior, and encodes them as scene-conditioned *rider tokens* to learn scene–trajectory compatibility. During training, NegWAM jointly optimizes the flow-matching and near-expert negative objectives, while the auxiliary rider branch is discarded at inference, incurring **no additional overhead**. On NAVSIM v2, NegWAM achieves **40.4 EPDMS on navhard**, a **4.1% relative improvement** over the strongest compared scorer-free WAM baseline, and **93.3 EPDMS on navtest**. These results demonstrate that NegWAM achieves strong performance in safety-critical scenarios while preserving routine-driving performance. Zero-shot closed-loop evaluation on HUGSIM further demonstrates cross-benchmark generalization.

## 🚀 Roadmap / Code Release

We will open-source the code step by step:

- [x] **Inference / evaluation code** — trajectory prediction & NAVSIM v2 evaluation scripts (`navtest` / `navhard two-stage`)
- [x] **Data preparation scripts** — preprocessing for NAVSIM v2
- [ ] **Pretrained checkpoints** — NegWAM weights for direct reproduction of reported results
- [ ] **Training code** — near-expert negative trajectory construction, repulsive-anchor flow-matching objective, and rider-token branch
- [ ] **HUGSIM closed-loop evaluation** — zero-shot closed-loop simulation scripts


## Getting Started

### Installation

Clone the repository and create the `negwam` environment from `environment.yml`. CUDA Toolkit 12.9 is required. `flash-attn` is installed after the environment exists, because it must be compiled against the PyTorch build in that file.

```shell
git clone https://github.com/Zizizi-hao/NegWAM
cd NegWAM

### 1. Create a conda environment
conda env create -f environment.yml
conda activate negwam

### 2. Install Flash Attention
export CUDA_HOME=/usr/local/cuda-12.9   # CUDA 12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
pip install flash-attn==2.8.3 --no-build-isolation
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

### Prepare Datasets

Follow the [NAVSIM installation guide](https://github.com/autonomousvision/navsim) to download the nuPlan maps and OpenScene data, then build the metric cache.

Put all data under `dataset/` and point `OPENSCENE_DATA_ROOT` at that directory. NAVSIM v2 supplies the raw logs, sensor blobs, maps, and the two-stage splits. The training and evaluation scripts read the cache directories produced from those files.

```shell
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="/path/to/NegWAM"
export OPENSCENE_DATA_ROOT="${NAVSIM_DEVKIT_ROOT}/dataset"
export NUPLAN_MAPS_ROOT="${OPENSCENE_DATA_ROOT}/maps"
export NAVSIM_EXP_ROOT="${NAVSIM_DEVKIT_ROOT}/exp"
```

Build the cache after the raw data is in place. Metric cache first, then negative trajectories, then features. Each script writes to `dataset/cache/` under the split name set inside the script (`navtrain` for training).

```shell
bash ./scripts/cache_data/run_feature_caching.sh
bash ./scripts/cache_data/run_metric_caching.sh
```

The overall structure of NegWAM:
```txt
NegWAM
├── exp/
├── navsim/
├── checkpoints/
├── scripts/
├── dataset/
│   ├── navsim/
│   │   ├── maps/
│   │   ├── navhard_two_stage/
│   │   ├── navsim_logs/
│   │   ├── sensor_blobs/
│   │   ├── warmup_two_state/
```

### Evaluation


```shell
### evaluation on navhard
bash ./scripts/evaluation/navhard/run_negwam_navhard_evaluation_multi_gpu.sh
### evaluation on navtest
bash ./scripts/evaluation/navtest/run_negwam_navtest_evaluation_multi_gpu.sh
```

## 📊 Important Results

### NAVSIM v2 — *navhard* Benchmark

![](https://ik.imagekit.io/zizizihao/NegWAM/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-11-05%20202432.png)Put all data under `dataset/` and point `OPENSCENE_DATA_ROOT` at that directory. NAVSIM v2 supplies the raw logs, sensor blobs, maps, and the two-stage splits. The training and evaluation scripts read the cache directories produced from those files.


### Analysis

<p align="center">
  <img src="https://ik.imagekit.io/zizizihao/NegWAM/rider_token.jpg" width="38%">
  <img src="https://ik.imagekit.io/zizizihao/NegWAM/scene_trajectory_com.jpg" width="58%">
</p>

## 🎬 Qualitative Results

### NAVSIM v2


<p align="center">
  <img src="https://ik.imagekit.io/zizizihao/NegWAM/figure_03_qualitative.jpg" width="95%">
</p>

### HUGSIM Zero-shot Closed-loop Evaluation

<table>
  <tr>
    <td align="center">
      <img src="https://ik.imagekit.io/zizizihao/NegWAM/hugsim_video_results_2.gif" width="100%">
      <br>Yielding to a left-side lane change
    </td>
    <td align="center">
      <img src="https://ik.imagekit.io/zizizihao/NegWAM/hugsim_video_results1.gif" width="100%">
      <br>Avoiding the hard braking of the leading vehicle
    </td>
  </tr>
</table>

## 🙏 Acknowledgements

We thank the authors of [NAVSIM](https://github.com/autonomousvision/navsim) and [HUGSIM](https://github.com/hyzhou-ryd/hugsim) for open-sourcing their benchmarks and evaluation tools, and the authors of [DriveWAM](https://github.com/chenshi3/DriveWAM), [BeyondDrive](https://github.com/wjl2244/BeyondDrive), and [GTRS](https://github.com/NVlabs/GTRS) for their open-source codebases, on which our implementation is built.

## 📄 License  and Citation

This project is licensed under the [Apache License 2.0](LICENSE).

If you find this work useful, please cite:
```bibtex
@misc{zhang2026negwam,
  title = {Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified World-Action Model},
  author = {Zihao Zhang and Xinyu Zhang and Bing Zeng and Xingyu Chen and Zhenning Li and Bin Yu and Ziyuan Pu},
  year = {2026},
  eprint = {2611.08111},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2611.08111}
}
```
