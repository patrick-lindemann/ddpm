# Denoising Diffusion Probabilistic Models 

This repository contains the code for the seminar paper "Denoising Diffusion Probabilistic Models" that was implemented as part of the course "Project Machine Learning" in WS23/24.

## Installation

```
pip install .
```

This will install the dependencies listed in [pyproject.toml](pyproject.toml).

## Usage

### Applying Noise

```
python3 apply_noise.py <image_path> [parameters]
```

This routine visualizes the forward diffusion process. It takes an image as parameter (e.g. one of the example images in `data/images`) and applies gradual noise according to the pre-configured schedule.

![Cosine Schedule](docs/images/schedule-cosine.png?raw=true "Cosine Schedule")

| Parameter | Description | Type | Default | Annotations |
| --- | --- | --- | --- | --- |
| `--schedule` |The schedule to use | linear, polynomial, cosine, sigmoid | linear | |
| `--schedule-steps` | The number of time steps for the diffusion process | int | 10 | |
| `--schedule-start` | The starting value for the schedule | float | 0.0 | |
| `--schedule-end` | The ending value for the schedule | float | 1.0 | |
| `--schedule-tau` | The tau value for the schedule | float | None | Only applicable for polynimial, cosine and sigmoid schedules |
| `--image-size` | The (quadratic) size to scale the image to | int | 128 | |
| `--outdir` | The directory to save the results to | pathlib.Path | `./out/forward/<schedule>`| |
| `--device` | The device to use | torch.Device | "cpu" | |
| `--verbose` | Enable verbose logging | bool | False | |

### Denoiser Training

```
python3 train_denoiser.py [parameters]
```

This routine traines the basic denoiser implementation on a specified dataset such as CIFAR-10, CelebA or LSUN. It outputs the trained model and a metadata file containing the training parameters for the test routine.

| Parameter | Description | Type | Default | Annotations |
| --- | --- | --- | --- | --- |
| `--dataset` | The name of the dataset to use during the training process. | cifar10, celeba, lsun | "cifar10" | |
| `--epochs` | The number of epochs to train for | int | 300 | |
| `--batch-size` | The batch size for the data loader | int | 16 | |
| `--schedule` |The schedule to use | str | "linear", "quadratic", "cosine", "sigmoid" | |
| `--schedule-steps` | The number of time steps for the diffusion process | int | 1000 | |
| `--schedule-start` | The starting value for the schedule | float | 0.0 | |
| `--schedule-end` | The ending value for the schedule | float | 1.0 | |
| `--schedule-tau` | The tau value for the schedule | float | None | Only applicable for polynimial, cosine and sigmoid schedules |
| `--train-size` | The size of the training set | float | 0.8 | |
| `--learning-rate` | The learning rate for the optimizer | float | 1e-3 | |
| `--outdir` | The directory to save the results to | pathlib.Path | `./out/train` | |
| `--device` | The device to use | cpu, cuda | cpu | |
| `--verbose` | Enable verbose logging | bool | False | |

## Denoiser Testing

```
python3 test_denoiser.py <model_dir> [parameters]
```

This routine is used to evaluate the trained denoiser. It takes the path to the model directory
(by default `out/train`) that contains the model parameters in `model.pt` and the training parameters in `metadata.json`. It then samples a random time step in [0, `<scheduler_steps>`] or uses the specified time step and evaluates the denoiser on a random image of the test set.

![Denoiser Result](docs/images/denoiser-result.png?raw=true "Denoiser Result")

| Parameter | Description | Type | Default | Annotations |
| --- | --- | --- | --- | --- |
| `--time-step` | The time step to evaluate the denoiser at | int | None | |
| `--device` | The device to use | torch.Device | "cpu" | |
| `--outdir` | The directory to save the results to | pathlib.Path | `./out/test`| |
| `--verbose` | Enable verbose logging | bool | False | |

#### Parameters

## Authors

- **Saif Daknou** - [Github](https://github.com/daknous)
- **Patrick Lindemann** - [Github](https://github.com/PatrickLindemann)
- **Nguyen Pham** - [Github](https://github.com/pdcnguyen)
