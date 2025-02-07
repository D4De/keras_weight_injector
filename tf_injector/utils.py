import importlib.resources
import os
import requests
import zipfile
import csv
from pathlib import Path
import shutil
from tensorflow import keras  # type:ignore
import tensorflow as tf

from PIL import Image
import numpy as np
from tqdm import tqdm


SUPPORTED_DATASETS = ["CIFAR10", "CIFAR100", "GTSRB"]

SUPPORTED_MODELS = [
    "ResNet18",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
    "ResNet1202",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "GoogLeNet",
    "MobileNetV2",
    "InceptionV3",
    "Vgg11_bn",
    "Vgg13_bn",
    "Vgg16_bn",
    "Vgg19_bn",
]

MODULE_PATH = Path(str(importlib.resources.files(__package__)))
DEFAULT_DATASET_PATH = MODULE_PATH / "../datasets"
DEFAULT_MODEL_PATH = MODULE_PATH / "../models"

INJECTED_LAYERS_TYPES = (keras.layers.Conv2D, keras.layers.Dense)

REPORT_HEADER = (
    "inj_id",
    "layer_weigths",
    "bit_pos",
    "n_injections",
    "top_1_correct",
    "top_5_correct",
    "top_1_robust",
    "top_5_robust",
    "masked",
    "non_critical",
    "critical",
)

DEFAULT_REPORT_DIR = MODULE_PATH / "../reports"


def _downloader(url, file_path):
    os.makedirs(file_path.parents[0], exist_ok=True)
    with open(file_path, "wb") as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            tqdm_params = {
                "total": total,
                "miniters": 1,
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=1024*32):
                    pb.update(len(chunk))
                    f.write(chunk)


def _extractor(file_path):
    with zipfile.ZipFile(file_path) as zipped:
        zipped.extractall(file_path.parents[0])


def create_tf_dataset(ds_path, out_path, labels_dict):
    """
    Since .ppm images are not supported by TF, converts them with Pillow then
    efficiently stores them in a TF dataset.
    """
    print("creating dataset", ds_path)
    imgs = []
    labels = []
    for file in tqdm(sorted(os.listdir(ds_path))):
        if file.endswith(".ppm"):
            img = Image.open(ds_path / file)
            img = img.resize((50, 50), resample=Image.Resampling.BILINEAR)
            b_img = np.asarray(img)
            rgb_img = b_img.view(dtype=np.uint8)  # useless?
            label = labels_dict[file]
            imgs.append(rgb_img)
            labels.append(label)

    labels = np.array([labels], dtype=np.uint8).T
    imgs = np.asarray(imgs)
    print(imgs.shape)
    print("packing into TF dataset...")
    dt = tf.data.Dataset.from_tensor_slices((imgs, labels))
    dt.save(str(out_path), compression="GZIP")
    print("done")


def download_gtsrb():
    """
    Downloads the GTSRB test dataset together with GT labels.
    The images are stored in a TF dataset.
    """
    image_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    gt_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
    gtsrb_path = DEFAULT_DATASET_PATH / "GTSRB"

    print("Downloading dataset...")
    _downloader(image_url, gtsrb_path / "dataset.zip")
    _downloader(gt_url, gtsrb_path / "gt.zip")
    print("done")

    print("extracting images...", end=" ")
    _extractor(gtsrb_path / "dataset.zip")
    _extractor(gtsrb_path / "gt.zip")
    print("done")

    # maps each file with its GT class
    file_class = {}
    with open(gtsrb_path / "GT-final_test.csv", "r") as f:
        reader = csv.reader(f, delimiter=";")
        next(iter(reader))
        for row in reader:
            file_class[row[0]] = int(row[-1])

    create_tf_dataset(
        gtsrb_path / "GTSRB/Final_Test/Images", gtsrb_path / "GTSRB_keras", file_class
    )
    os.remove(gtsrb_path / "dataset.zip")
    os.remove(gtsrb_path / "gt.zip")
    os.remove(gtsrb_path / "GT-final_test.csv")
    shutil.rmtree(gtsrb_path / "GTSRB")
