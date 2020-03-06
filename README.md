# Generation3D

3D Shape Generation Baselines in PyTorch.

![](https://img.shields.io/static/v1?label=Gen3D&message=0.1.1&color=blue) 
![](https://img.shields.io/static/v1?label=PyTorch&message=1.3.1&color=orange)
![](https://img.shields.io/static/v1?label=CUDA&message=10.0&color=green)
#### Feature

- Hack of DataParallel for balanced memory usage
- More Models **WIP**
- Configurable model parameters
- Customizable model, dataset

#### Representation

- 💎 Polygonal Mesh
- 👾 Volumetric
- 🎲 Point Cloud
- 🎯 Implicit Function
- 💊 Primitive

#### Input Observation
- 🏞 RGB Image
- 📡 Depth Image
- 👾 Voxel
- 🎲 Point Cloud
- 🎰 Unconditional Random

#### Evaluation Metrics

- Chamfer Distance
- F-score
- IoU

#### Model Zoo
- [x] 💎 Pixel2Mesh
- [x] 🎯 DISN
- [x] 👾 3DGAN
- [ ] 👾 Voxel Based Method
- [ ] 🎲 PointCloud Based Method

## Get Started

### Environment
- Ubuntu 16.04 / 18.04
- Pytorch 1.3.1
- CUDA 10
- conda > 4.6.2

Using Anaconda to install all dependences.

```
conda env create -f environment.yml
```

### Train

```
CUDA_VISIBLE_DEVICES=<gpus> python train.py --options <config>
```

### Predict

```
CUDA_VISIBLE_DEVICES=<gpus> python predictor.py --options <config>
```

### Evaluation [WIP]

### Custom guide

- custom scheduler for `training/inference` loop, add code in `scheduler` and inherit base class.
- custom model in `models/zoo`
- custom config options in `utils/config`
- custom dataset in `datasets/data`

### External
- Chamfer Distance

## Baselines

### Pixel2Mesh 🏞 💎

- Input: RGB Image 
- Representation: Mesh
- Output: Mesh <sub>camera-view</sub>

### DISN 🏞 🎯

- Input: RGB Image
- Representation: SDF
- Post-processing: Marching Cube
- Output: Mesh <sub>camera-view</sub>

### 3DGAN 🎰 👾

- Input: Random Noise
- Representation: Volumetric
- Output: Voxel

---

#### Acknowledgements

Our work is based on the codebase of [an unofficial pixel2mesh framework](https://github.com/noahcao/Pixel2Mesh). The Chamfer loss code is based on [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch).

Official baseline code

- [DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction](https://github.com/Xharlie/DISN)

- [Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images](https://github.com/nywang16/Pixel2Mesh)

- [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://github.com/zck119/3dgan-release)

#### License

Please follow the License of official implementation for each model.
