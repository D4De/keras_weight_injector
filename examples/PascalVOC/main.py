import sys
# We do the assumption that tf_injector is not installed in your system. 
# To make it work anyways, we manually add the path of the injector to the PYTHONPATH
# For more info: https://docs.python.org/3/using/cmdline.html#environment-variables
sys.path.append("../../")
import gg as kwi 

from tqdm.auto import tqdm

import os
import numpy as np
import tensorflow as tf

from VOCDataset import *
keras = tf.keras

HEIGHT = 520
WIDTH = 520

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
def transform_np(x, mean=mean, std=std):
    # transform in cupy compatible
    """ x = tf.experimental.dlpack.to_dlpack(x)
    x = cp.from_dlpack(x) """
    
    # normalize between 0 and 1
    x = x - x.min()
    x = x / x.max()

    # normalize with 
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    # tf.keras.layers.Normalization(mean = [0.485, 0.456, 0.406], variance = [0.229, 0.224, 0.225])
    x = np.divide(
            np.subtract(x, mean),
            np.maximum(std, 1e-7)
        )

    # return to tf from cupy
    """ x = x.toDlpack()
    x = tf.experimental.dlpack.from_dlpack(x) """
    return x

def transform_tf(x, mean=mean, std=std):
    x = x - tf.reduce_min(x)
    x = x / tf.reduce_max(x)
    x = tf.divide(
            tf.subtract(x, mean),
            tf.cast(tf.maximum(std, 1e-7), tf.float32)
    )
    x = tf.image.resize(x, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    return x

def target_transform(x):
    x = tf.expand_dims(x, -1) # we need this dim to use tf.image.resize
    #resize
    x = tf.image.resize(x, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.squeeze(x)
    return x

dataset = VOCDataset(
    root_dir="VOCdevkit/VOC2012/",
    which_split="val",
    classes = VOCDataset.PASCAL_CLASSES,
    transform = transform_tf,
    target_transform = target_transform,
    HEIGHT = HEIGHT, WIDTH = WIDTH,
)

numclass = 21
batch_size = 16
dataloader = Dataloder(dataset, batch_size=batch_size)

model = keras.models.load_model("DeepLabV3.keras")
model.trainable = False

def get_label(out):
    out = out[0]
    out = tf.argmax(out, axis=-1)
    return tf.cast(out, tf.uint8)

def cast_labels_to_uint8(labels): 
    return tf.cast(labels, tf.uint8)

# prepare the injector
injector = kwi.Injector(
    model,
    dataloader,
    transform_output = get_label,
    transform_label = cast_labels_to_uint8,
)

# load the fault list 
injector.load_fault_list("./keras_fault_list.csv")

#validate the fault list
if len(sys.argv) > 1 and sys.argv[1] == "validate":
    print("Validating fault list...")
    injector.validate()
    print("fault list validated!!!")
else:
    print("Skipping fault list validation...")

# run the campaign
with kwi.CampaignWriter(
    dataloader,
    "DeepLabV3+",
    (
        "inj_id",
        "target_layer",
        "layer_weigths",
        "bit_pos",
        "input_id",
        "n_injections",
        "pixel_accuracy_golden",
        "pixel_accuracy_label",
        *(f"IOU_golden_{i}" for i in range(21)),
        *(f"IOU_label_{i}" for i in range(21)),
    ),
    "./reports",
    one_line_per_input = True,
) as cw:
        injector.run_campaign(
            batch= batch_size,
            metrics =  [
                kwi.metrics.PixelAccuracyMetric,
                kwi.metrics.ImageIntersectionOverUnionMetric(21),
            ],
            outputter=cw,
            save_scores=False,
            metrics_on_labels=True,
        )
