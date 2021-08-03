"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import time

th.set_num_threads(1)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    print('args dict: ',args_to_dict(args, model_and_diffusion_defaults().keys()))
    print('args clip denoised: ', args.clip_denoised)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    start = time.time()

    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}

        if args.class_cond:
            y = np.full((1,144), -2, dtype=np.float32)
            y[0][0] = -1
            y[0][1] = 0.9
            y[0][2] = 0.8

            y[0][3] = 0.9
            y[0][4] = -1
            y[0][5] = 0.1

            y[0][6] = 0.8
            y[0][7] = 0.1
            y[0][8] = -1


            z = np.full((1,16,3), -2, dtype=np.float32)
            print(z.shape)
            z[0][0][0] = 1
            z[0][0][1] = -1
            z[0][0][2] = -1

            z = np.transpose(z, (0, 2, 1))
            model_kwargs["y"] = th.from_numpy(y)
            model_kwargs["z"] = th.from_numpy(z)

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs
        )
        print(sample)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 1)
        sample = sample.contiguous()
        print(sample)
        print('generation time: ', time.time()-start)
        start = time.time()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
