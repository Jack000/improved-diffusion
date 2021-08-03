from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle
import random

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    #all_files = [i for i in range(10000)]
    classes = None
    #if class_cond:
        # file names are formatted as blahblah.label.svg.txt
    #    class_names = [bf.basename(path).split(".")[-3] for path in all_files]
    #    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #    classes = [sorted_classes[x] for x in class_names]

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["txt", "jpg", "jpeg", "png", "gif","pkl"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        
        text = ''
        with open(path, 'rb') as infile:
            image_data = pickle.load(infile)

        np_image = np.zeros((16,3), dtype=np.float32)
        pal = image_data[0]
        for i, rgb in enumerate(pal):
            np_image[i][0] = (rgb[0]/127.5)-1 # range -1 to 1
            np_image[i][1] = (rgb[1]/127.5)-1
            np_image[i][2] = (rgb[2]/127.5)-1

        out_dict = {}
        cond = image_data[1]
        y = np.full(144, -2, dtype=np.float32)
        for i, c in enumerate(cond):
            y[i] = (c/100) + np.random.normal(0, 0.01) # range -1 to 1 plus a bit of noise

        z = np.full((16,3), -2, dtype=np.float32)

        if random.randint(0,100) < 50:
            out_dict["z"] = np.transpose(z, [1, 0])
        else:
            randlist = list(range(len(pal)))
            randnum = random.randint(1, len(pal)-1)
            random.shuffle(randlist)
            randlist = randlist[:randnum]

            for i, rgb in enumerate(pal):
                if i in randlist:
                    z[i][0] = (rgb[0]/127.5)-1 # range -1 to 1
                    z[i][1] = (rgb[1]/127.5)-1
                    z[i][2] = (rgb[2]/127.5)-1

            out_dict["z"] = np.transpose(z, [1, 0])

        out_dict["y"] = y
        np_image = np.transpose(np_image, [1, 0])

        return np_image, out_dict
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        #while min(*pil_image.size) >= 2 * self.resolution:
        #    pil_image = pil_image.resize(
        #        tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        #    )

        #scale = self.resolution / min(*pil_image.size)
        #pil_image = pil_image.resize(
        #    tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        #)

        #arr = np.array(pil_image.convert("RGB"))
        #crop_y = (arr.shape[0] - self.resolution) // 2
        #crop_x = (arr.shape[1] - self.resolution) // 2
        #arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        #arr = arr.astype(np.float32) / 127.5 - 1


        #return np.random.randn(2,512).astype(np.float32), {"y":np.array(self.local_classes[idx], dtype=np.int64)}


        #out_dict = {}
        #if self.local_classes is not None:
            #out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        #return np.transpose(arr, [2, 0, 1]), out_dict
