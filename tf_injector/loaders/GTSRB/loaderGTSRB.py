'''
Loading file for GTSRB
'''
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

import sys

DEFAULT_DATASET_PATH = Path("tf_injector/loaders/GTSRB/dataset")



# DAWNLOAD DATASET

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
    gtsrb_path = DEFAULT_DATASET_PATH / "GTSRB_keras"

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

def preprocess(dataset : tf.data.Dataset) -> tf.data.Dataset:
    """
    Preprocesses the dataset by normalizing the images and converting the labels to one-hot encoding.
    """
    def preprocess_image(image, label):
        mean = tf.constant([0.3403, 0.3121, 0.3214], dtype=tf.float32)
        std = tf.constant([0.2724, 0.2608, 0.26690], dtype=tf.float32)

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image - mean) / std
        
        return image, label

    dataset = dataset.map(preprocess_image)
    return dataset

def load_gtsrb():
    dt_path = DEFAULT_DATASET_PATH / "GTSRB_keras" / "GTSRB_keras"
    if not os.path.exists(dt_path):
        download_gtsrb()
    dt = tf.data.Dataset.load(str(dt_path), compression="GZIP")
    return preprocess(dt)

