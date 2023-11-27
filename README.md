# Denoising Diffusion Probabilistic Models 

## Installation

```
pip install .
```

## Usage

### Applying Noise

```
apply_noise.py <image_path> [parameters]
```

#### Parameters

| Parameter | Description | Type | Default | Annotations |
| --- | --- | --- | --- | --- |
| `--device` | The device to use | torch.Device | "cpu" | |
| `--time-steps` | The number of time steps | int | 10 | |
| `--schedule` |The schedule to use | str | "linear", "quadratic", "cosine", "sigmoid" | |
| `--schedule-start` | The starting value for the schedule | float | 0.0 | |
| `--schedule-end` | The ending value for the schedule | float | 1.0 | |
| `--schedule-tau` | The tau value for the schedule | float | None | Only applicable for polynimial, cosine and sigmoid schedules |
| `--image-size` | The (quadratic) size to scale the image to | int | 128 | |
| `--outdir` | The directory to save the results to | pathlib.Path | "./out/" | |
| `--export-all` | Export all images, not only the visualization | bool | False | |
| `--verbose` | Enable verbose logging | bool | False | |


### Denosing Training

```
train_denoiser.py [parameters]
```

#### Parameters

## Authors

- **Saif Daknou** - [Github](https://github.com/daknous)
- **Patrick Lindemann** - [Github](https://github.com/PatrickLindemann)
- **Nguyen Pham** - [Github](https://github.com/pdcnguyen)
