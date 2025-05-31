import numpy as np
import dummy_dataset as utils 
import tensorflow as tf

# We do the assumption that tf_injector is not installed in your system. 
# To make it work anyways, we manually add the path of the injector to the PYTHONPATH
# For more info: https://docs.python.org/3/using/cmdline.html#environment-variables
import sys
sys.path.append("../../")
import gg as kwi 

keras = tf.keras

# Dummy model 
inp = tf.keras.Input(shape=(520, 520, 3))
x = keras.layers.Conv2D(21, (3,3), padding="same")(inp)
model = tf.keras.models.Model(inp, [x,x])

model.trainable=False

print(model.summary())

dataset = utils.VOCDataset()
dataloader = utils.Dataloder(dataset, batch_size=16)

def get_label(out):
    out = out[0]
    out = tf.argmax(out, axis=-1)
    return tf.cast(out, tf.uint8)

def label_to_uint8(label):
    return tf.cast(label, tf.uint8)

# prepare the injector
injector = kwi.Injector(
    model,
    dataloader,
    transform_output = get_label,
    transform_label = label_to_uint8,
)

# load the fault list 
injector.load_fault_list("dummy_keras_fault_list.csv")

#validate the fault list
injector.validate()

batch_size = 2

# run the campaign
with kwi.CampaignWriter(
    dataloader,
    "DeepLabV3+",
    (
        "inj_id",
        "target_layer",
        "input_id",
        "layer_weigths",
        "bit_pos",
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
            metrics = [
                kwi.metrics.PixelAccuracyMetric,
                kwi.metrics.ImageIntersectionOverUnionMetric(21),
            ],
            outputter=cw,
            save_scores=False,
            metrics_on_labels=True,
        )
