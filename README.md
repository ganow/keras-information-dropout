keras-information-dropout
=========================

> Achille, A., & Soatto, S. (2016). *Information Dropout: learning optimal representations through noise.* arXiv:1611.01353

[Keras](https://keras.io/) implementation of the Information Dropout ([arXiv:1611.01353](https://arxiv.org/abs/1611.01353)) paper.

![generated image](vae_information_dropout/images/generated_image.png)

![latent space interpolation](vae_information_dropout/images/latent.png)

![weight diagram](vae_information_dropout/images/weight_diagram.png)

## Usage

### Generate dataset

```bash
$ luajit download_mnist.lua
$ luajit make_cluttered_dataset.lua
$ python t7_to_hdf5.py
```

### Training

```bash
```

