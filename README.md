# improved-diffusion

This is the DDPM model for huemint.com, based on https://github.com/openai/improved-diffusion

The main change from the original code:

- use 1 dimension instead of 2
- load numpy arrays instead of image files
- replaced category embedding with linear layer (y keyword argument)
- add additional input to UNET for conditional generation (z keyword argument)

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

## Preparing Data

The modified training code reads numpy arrays from a directory of pkl files.

For creating your own dataset, simply pickle.dump() each numpy array into a directory with a .pkl extension. The contents of the .pkl should be a tuple, with data[0].shape = (16,3) (for the color palette) and data[1].shape = (144) (flattened contrast matrix)

## Training

Here are some example training parameters:

```
MODEL_FLAGS="--image_size 16 --num_channels 128 --num_res_blocks 3 --learn_sigma True --class_cond True --dropout 0.3 --dims 1"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 2e-4 --batch_size 6400"

python scripts/image_train_hue.py --data_dir /path/to/data/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

## Sampling

A sampling script is included for testing purposes. Note: there is an unidentified memory leak in the original code that appears during inference. Using this code as-is in a server setting is not recommended.

```
python scripts/image_sample_hue.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```