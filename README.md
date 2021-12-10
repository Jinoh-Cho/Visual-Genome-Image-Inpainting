# Visual-Genome-Image-Inpainting


This repository aims to train an autoEncoder-based BigGAN model for image inpainting. Specifically, the autoencoder-based BigGAN model tries to fill a missing part of an image using visual common sense.

# Requirements

First, install PyTorch meeting your environment (at least 1.7, recommmended 1.10):
```bash
pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Then, use the following command to install the rest of the libraries:
```bash
pip3 install tqdm ninja h5py kornia matplotlib pandas sklearn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg prdc
```

# Quick Start

Before starting, users should login wandb using their personal API key.

```bash
wandb login PERSONAL_API_KEY
```

* Train (``-t``) and evaluate (``-e``) the BigGAN-style autoencoder model using GPU ``0,1,2,3``.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -e -cfg "./src/configs/VisualGenome/BigGAN.yaml" -data DATA_PATH -save SAVE_PATH
```

Try ``python3 src/main.py`` to see available options.

## Dataset

make the folder structure of the dataset as follows:

```
data
└── VisualGenome
    ├── train
       ├── source
       │   ├── train0.png
       │   ├── train1.png
       │   └── ...
       └── target
           ├── train0.png
           ├── train1.png
           └── ...
    ```

## License
This Library is an open-source library under the MIT license (MIT). However, portions of the library are avaiiable under distinct license terms: Synchronized batch normalization is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/sync_batchnorm/LICENSE), HDF5 generator is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/hdf5.py), and differentiable SimCLR-style augmentations is licensed under [MIT license](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/simclr_aug.py).
