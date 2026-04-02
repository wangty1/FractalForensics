# 📖 FractalForensics: Proactive Deepfake Detection and Localization via Fractal Watermarks

Source code implementation of our paper [[arXiv](https://arxiv.org/abs/2504.09451)] accepted to Proceedings of the 33rd ACM International Conference on Multimedia (MM 2025 Oral).

## 📢 Updates

- [x] Open project page.
- [x] Update project page information.
- [x] Update arXiv version.
- [ ] Release inference code.
- [ ] Release trained weights.

## 🗂️ Datasets Used
FractalForensics is trained using CelebA-HQ and tested on CelebA-HQ and LFW. We do not own the datasets, and they can be downloaded from the official webpages.

* [Download CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [Download LFW](https://vis-www.cs.umass.edu/lfw/)

After splitting the image data following the official document of CelebA-HQ, the folder should be named as ```dataset_celeba_hq/``` and placed under ```image_data/```. For the cross-dataset evaluation under a balanced ratio, LFW is processed such that one for each identity is adopted. The directory should look like the following:
```
FractalForensics
└── image_data/
    ├── dataset_celeba_hq/
    │   ├── train/
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── 1000.jpg
    │   │   └── ...         
    │   └── test/
    │       ├── 10008.jpg
    │       └── ...    
    └── dataset_lfw/
        └── test/
            ├── AJ_Cook_0001.jpg
            └── ...
```

In this project, we prepared fractal watermarks with sufficient randomness for efficient loading and running. We provide the watermarks that we used for testing, which can be found in ```watermark_data/```. The watermarks are stored in a single ```.npy``` file. 

## 🔮 Test the Model

The model is tested following the configuration files located in ```configuration/```.

We use ```configuration/test_common.json``` to test the watermarks against all benign manipulations and derive the watermark recovery accuracies. We use ```configuration/test_deepfake.json``` to test the watermarks against Deepfake manipulations and derive the watermark recovery accuracies. 

We provide watermarks randomly generated following the variations in advance, under ```FractalForensics/data_process/watermark_cache```, for direct loading when testing the model. 

We use ```main.py``` to test the model against potential adversaries. 

## 🎭 Deepfake Models Used

FractalForensics is tested in a black box manner against eight Deepfake models. Since we don't own the source code, we recommend downloading and placing the model source code and weights by yourself. Unzipped folders of the models should be placed under ```model/``` folder so that the classes in ```model/deepfakes.py``` can utilize the generative models. 

The source code can be found at the following links:
* [SimSwap (ACM MM 2020)](https://github.com/neuralchen/SimSwap)
* [InfoSwap (CVPR 2021)](https://github.com/GGGHSL/InfoSwap-master)
* [UniFace (ECCV 2022)](https://github.com/xc-csc101/UniFace)
* [E4S (CVPR 2023)](https://github.com/e4s2022/e4s/tree/main)
* [DiffSwap (CVPR 2023)](https://github.com/wl-zhao/DiffSwap)
* [StarGAN (CVPR 2020)](https://github.com/clovaai/stargan-v2)
* [StyleMask (FG 2023)](https://github.com/StelaBou/StyleMask)
* [HyperReenact (ICCV 2023)](https://github.com/StelaBou/HyperReenact)

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 📝 Citation
If you find our work useful, please properly cite the following:
```
@inproceedings{Wang2025FractalForensics,
author = {Wang, Tianyi and Cheng, Harry and Liu, Ming-Hui and Kankanhalli, Mohan},
title = {FractalForensics: Proactive Deepfake Detection and Localization via Fractal Watermarks},
year = {2025},
isbn = {9798400720352},
doi = {10.1145/3746027.3754544},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {7210–7219}
}
```
