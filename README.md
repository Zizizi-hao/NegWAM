<div align="center">

# Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified World-Action Model

**Zihao Zhang<sup>1,2</sup> · Xinyu Zhang<sup>3</sup> · Bing Zeng<sup>3</sup> · Xingyu Chen<sup>4</sup> · Zhenning Li<sup>5</sup> · Bin Yu<sup>6</sup> · Ziyuan Pu<sup>3</sup>**

<sub><sup>1</sup> School of Cyber Science and Engineering, Southeast University &nbsp;·&nbsp; <sup>2</sup> Shenzhen Loop Area Institute &nbsp;·&nbsp; <sup>3</sup> School of Transportation, Southeast University</sub>
<br>
<sub><sup>4</sup> School of Mechanical Engineering, Guangxi University &nbsp;·&nbsp; <sup>5</sup> University of Macau &nbsp;·&nbsp; <sup>6</sup> School of Transportation Science and Engineering, Beihang University</sub>
<br>
</div>

---

<p align="center">
  <img src="https://ik.imagekit.io/zizizihao/NegWAM/figure_02_methods.jpg" width="95%">
</p>


World-action models (WAMs) jointly model future traffic states and driving actions for end-to-end autonomous driving in intelligent transportation systems. However, their action learning remains dominated by expert-trajectory imitation, leaving unsafe alternatives close to expert trajectories insufficiently constrained. To improve WAM robustness in safety-critical scenarios, we propose **NegWAM**, a safety post-training framework for unified WAMs that learns explicit safety boundaries from *near-expert failure trajectories*. NegWAM constructs negative trajectories that are geometrically close to the expert trajectory yet carry safety risks, uses them as repulsive anchors to steer predictions away from dangerous behavior, and encodes them as scene-conditioned *rider tokens* to learn scene–trajectory compatibility. During training, NegWAM jointly optimizes the flow-matching and near-expert negative objectives, while the auxiliary rider branch is discarded at inference, incurring **no additional overhead**. On NAVSIM v2, NegWAM achieves **40.4 EPDMS on navhard**, a **4.1% relative improvement** over the strongest compared scorer-free WAM baseline, and **93.3 EPDMS on navtest**. These results demonstrate that NegWAM achieves strong performance in safety-critical scenarios while preserving routine-driving performance. Zero-shot closed-loop evaluation on HUGSIM further demonstrates cross-benchmark generalization.



## 🚀 Roadmap / Code Release

We will open-source the code step by step:

- [ ] **Inference / evaluation code** — trajectory prediction & NAVSIM v2 evaluation scripts (`navtest` / `navhard two-stage`)
- [ ] **Pretrained checkpoints** — NegWAM weights for direct reproduction of reported results
- [ ] **Training code** — near-expert negative trajectory construction, repulsive-anchor flow-matching objective, and rider-token branch
- [ ] **HUGSIM closed-loop evaluation** — zero-shot closed-loop simulation scripts
- [ ] **Data preparation scripts** — preprocessing for NAVSIM v2 / HUGSIM

## 📊 Important Results

### NAVSIM v2 — *navhard* Benchmark

<p align="center">
  <img src="https://ik.imagekit.io/zizizihao/NegWAM/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202026-11-05%20202432.png" width="95%">
</p>


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

We thank the authors of [NAVSIM](https://github.com/autonomousvision/navsim) and [HUGSIM](https://github.com/hyzhou-ryd/hugsim) for open-sourcing their benchmarks and evaluation tools, and the authors of [DriveWAM](https://github.com/chenshi3/DriveWAM)， [BeyondDrive](https://github.com/wjl2244/BeyondDrive), and [GTRS](https://github.com/NVlabs/GTRS) for their open-source codebases, on which our implementation is built.

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
