# Denoising Diffusion Probabilistic Models 

This repository contains the code for the seminar paper "Denoising Diffusion Probabilistic Models" that was implemented as part of the course "Project Machine Learning" in WS23/24. The reports for the three milestones can be found in the [docs/reports](docs/reports) directory.

![DDPM Paper](docs/images/paper.png?raw=true "DDPM Paper")

## Installation

```
pip install . -r requirements.txt
```

This will install the dependencies listed in [requirements.txt](requirements.txt).

## Training

```
python3 train_denoiser.py <dataset_name> [options]
```

Train the denoising module on one of the provided datasets, i.e.
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [MNIST](http://yann.lecun.com/exdb/mnist/), [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) or [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The dataset is split into a training set, which is used to train the denoiser, and a test set, which is used to validate the model at end of each epoch. After the training is complete, the training results (metadata, losses, etc.) are stored together with the model weights in a directory named after the experiment.

![Train and Test Losses](docs/images/train-test-losses.png?raw=true "Train and Test Losses")

### Positional Parameters

| Parameter | Description | Type | Annotations |
| --- | --- | --- | --- |
| `dataset` | The name of the dataset to use. | "cifar10" \| "mnist" \| "fgvc-aircraft" \| "flowers102" \| "celeba" | |

### Options

| Parameter | Description | Type | Default | Annotations |
| --- | --- | --- | --- | --- |
| `--run-name` | The name of the experiment. If not provided, the name will be generated automatically. | str | None | Generated names have the pattern `$timestamp`-`$dataset`. |
| `--image-size` | The (square) sample size of the images. | int | 32 | If the provided image size is different from the dataset's image size, the images will be up- or downsized accordingly. |
| `--time-steps` | The number of time steps for the diffusion process. | int | 1000 | |
| `--schedule` | The schedule type to use. | "linear" \|  "polynomial" \| "cosine" \|  "sigmoid" | "linear" | |
| `--schedule-start` | The start value for the schedule within [0,1]. | float | 0.0001 | |
| `--schedule-end` | The end value for the schedule within [0,1]. | float | 0.2 | |
| `--schedule-tau` | The tau value for the schedule | float \| None | None | Only applicable for `polynimial`, `cosine` and `sigmoid` schedules. |
| `--epochs` | The number of epochs to train for. | int | 1000 | |
| `--subset-size` | The number of samples to use from the dataset. If not provided, the entire dataset is used. | int \| None | None | |
| `--train-split` | The percentage in [0, 1] of the dataset to use for training. | float | 0.8 | |
| `--disable-validation` | Disable validation on the test set during training. | bool | False | |
| `--batch-size` | The batch size for the data loader. | int | 16 | |
| `--learning-rate` | The learning rate for the optimizer. | float | 2e-4 | |
| `--dropout-rate` | The dropout rate for the model. | float | 0.0 | |
| `--seed` | The random seed to use. If not specified, the randomized calculations will be non-deterministic. | int \|  None | None | |
| `--out-dir` | The directory to save the results to. | pathlib.Path | `./out/train` | |
| `--device` | The device to use. | "cpu" \|  "cuda" \| None | None | If not specified, "cuda" will be used if it is available. |
| `--verbose` | Flag for enabling verbose logging .| bool | False | |

## Image Generation

```
python3 generate_images.py <run_dir> <image_count> [options]
```

Generate images using a trained denoiser. The routine takes the path of a run created by `train_denoiser.py` and the number of images to generate. The resulting images are saved to the specified output directory.

![Generated CIFAR-10 Images](docs/images/cifar10-32x32.png?raw=true "Generated CIFAR-10 Images")

### Parameters

| Parameter | Description | Type | Annotations |
| --- | --- | --- | --- |
| `run_dir` | The directory containing the trained model. | Path | |
| `image_count` | The number of images to generate. | int | |


### Options

| Parameter | Description | Type | Default | Annotations |
| --- | --- | --- | --- | --- |
| `--batch-size` | The batch size for generating images. Must be a divisor of `num_images`. | int | None | |
| `--time-steps` | The number of time steps for the diffusion process. If not specified, number of time steps from the training will be used. | int \| None | None | |
| `--seed` | The random seed to use. If not specified, the randomized calculations will be non-deterministic. | int \|  None | None | |
| `--out-dir` | The directory to save the results to. If not specified, the images are saved to `<run_dir>/samples`. | pathlib.Path | None | |
| `--device` | The device to use. | "cpu" \|  "cuda" \| None | "cuda" | If not specified, "cuda" will be used if it is available. |


## Image Evaluation

```
python3 evaluate_images.py <image_dir> [options]
```

Calculate the [Inception Score](https://en.wikipedia.org/wiki/Inception_score) to evaluate the quality of generated images using the InceptionV3 model. The routine takes the path of a directory containing the generated images and saves their resulting score to `inception_score.txt` in the same directory.

### Positional Parameters

| Parameter | Description | Type | Annotations |
| --- | --- | --- | --- |
| `image_dir` | The directory containing the generated images. | Path | | |

### Options

| Parameter | Description | Type | Default | Annotations |
| --- | --- | --- | --- | --- |
| `--out-file` | The file to save the inception score to. If not specified, the score is saved to `<image_dir>/inception_score.txt`. | pathlib.Path | None | |
| `--device` | The device to use. | "cpu" \|  "cuda" \| None | "cuda" | If not specified, "cuda" will be used if it is available. |
| `--verbose` | Enable verbose logging. | bool | False | |

## Noise Schedule Visualization

```
python3 apply_noise.py <image_path> [options]
```

This routine visualizes the forward diffusion process. It takes an image as parameter (e.g. one of the example images in `data/images`) and applies gradual noise according to the pre-configured schedule.

![Cosine Schedule](docs/images/schedule-cosine.png?raw=true "Cosine Schedule")

### Positional Parameters

### Options

| Parameter | Description | Type | Default | Annotations |
| --- | --- | --- | --- | --- |
| `--schedule` |The schedule to use | linear, polynomial, cosine, sigmoid | linear | |
| `--schedule-steps` | The number of time steps for the diffusion process | int | 10 | |
| `--schedule-start` | The starting value for the schedule | float | 0.0 | |
| `--schedule-end` | The ending value for the schedule | float | 1.0 | |
| `--schedule-tau` | The tau value for the schedule | float | None | Only applicable for polynimial, cosine and sigmoid schedules |
| `--image-size` | The (quadratic) size to scale the image to | int | 128 | |
| `--outdir` | The directory to save the results to | pathlib.Path | `./out/forward/<schedule>`| |
| `--device` | The device to use | torch.Device | "cpu" | If not specified, "cuda" will be used if it is available. |
| `--verbose` | Enable verbose logging | bool | False | |


## Authors

- **Saif Daknou** - [Github](https://github.com/daknous)
- **Patrick Lindemann** - [Github](https://github.com/PatrickLindemann)
- **Nguyen Pham** - [Github](https://github.com/pdcnguyen)
