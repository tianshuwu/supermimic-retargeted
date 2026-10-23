# Sources and attribution

This is a derivative robot adaptation, not an original implementation of SuperMimic.

| Source | Role | Pinned revision |
|---|---|---|
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | Main training implementation; MIT copyright retained in LICENSE | `d9be385b3eeb13a638fb33fb200d3bfd9be9839f` |
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | Reference implementation consulted | `a1508d42c9d1699a52e8233ab16d08c488441ab9` |
| [NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter) | Retargeting and robot descriptions; separate Apache-2.0 project with third-party notices | `cbdb2b494a2af82137bf9b6f15122d60167281c4` |
| [AgibotTech/sonic_for_a3](https://github.com/AgibotTech/sonic_for_a3) | A3 motor configuration reference; external Apache-2.0 project | `c9df753d31ef63c930b825b1b6767cabd37c1549` |
| [meetsitaram/sonic-x2](https://github.com/meetsitaram/sonic-x2) | X2 motor configuration reference, fetched separately | `c9959443f80276533083a8b31ecd3760a41eaa65` |
| [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | Simulation framework and G1 asset | `8bb2bc9e187797689ec82c78c35019afd2fd0376` |

We added embodiment-dependent dimensions, robot profiles, LAFAN skeleton/frame/pelvis adaptation, contact QC, replay balancing, and shared-scene ghost videos. AI-assisted coding was used throughout. The algorithm and most training code originate from the projects above. Original upstream documentation remains available in those repositories.

Original paper: **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480).

Robot descriptions, meshes, datasets and third-party model weights are not redistributed here. Fetch them from the original projects and retain their licenses/notices; this repository's MIT license does not replace those terms. X2 reference files are fetched externally rather than relicensed as our own code.

LAFAN1 source: [Ubisoft](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), revision `94084601bacdf9cc3764b5c73daaeccae6035fac`. G1 retargeted CSV source used in this experiment: [ModelScope](https://www.modelscope.cn/datasets/AI-ModelScope/LAFAN1_Retargeting_Dataset), revision `88771b167b7c5b87896fbef6799b3ae8150450e8`.

BONES-SEED source: BONES Studio, `bones-studio/seed`. The optional GitCode mirror used for this run is pinned at `3347a32c4542a0a1165d6c4667ef631df2bc4c9d`. Dataset terms remain those of the dataset provider.

Published manifest names and selection metadata retain their dataset-source terms; they are not a relicensing of the datasets.
