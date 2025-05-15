import tensorflow as tf
import numpy as np
import os

'''
uses tf_from_tensor_slices instead of PIL to load images
'''


# fixe params
PATH = "/Users/domenicopalumbo/Documents/VOCdevkit/VOC2012/"
batch_size = 32

HEIGHT = 520
WIDTH = 520

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

PASCAL_COLORS=[
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
PASCAL_CLASSES = ['background',
                      'aeroplane',
                      'bicycle',
                      'bird',
                      'boat',
                      'bottle',
                      'bus',
                      'car',
                      'cat',
                      'chair',
                      'cow',
                      'diningtable',
                      'dog',
                      'horse',
                      'motorbike',
                      'person',
                      'pottedplant',
                      'sheep',
                      'sofa',
                      'train',
                      'tvmonitor'
                     ]


def transform_tf(x : tf.Tensor, mean=mean, std=std):
    # (None, None, 3)
    x = x - tf.reduce_min(x)
    x = x / tf.reduce_max(x)
    x = tf.divide(
            tf.subtract(x, mean),
            tf.cast(tf.maximum(std, 1e-7), tf.float32)
    )
    x = tf.image.resize(x, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    return x

def target_transform(x : tf.Tensor):
    # (None, None, 3)
    #x = tf.expand_dims(x, -1)
    #(None, None, 3, 1)
    x = tf.image.resize(x, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # (None, 520, 520, 1)
    #x = tf.squeeze(x)
    # (None, 520, 520) -> (366, 520, 520)
    return x

def load_preprocess_image(image_path: tf.Tensor) -> tf.Tensor:
    img_contents = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = transform_tf(img)
    return img

def load_preprocess_mask(mask_path: tf.Tensor) -> tf.Tensor:
 
    mask_contents = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_contents, channels=3)

    mask = target_transform(mask)

    # map each elemnent of the mask to its class using the PASCAL_COLORS
    COLOR_LABEL_MAP = {
        tuple(PASCAL_COLORS[i]): i for i in range(len(PASCAL_COLORS))
    }
    

    def convert_rgb_to_class_indices(mask_tensor : tf.Tensor) -> np.ndarray:
        '''
        [R, G, B] -> index
        (520, 520, 3) -> (520, 520)
        '''
        mask_np = mask_tensor.numpy()
        label_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int32)
        
        # Per ogni pixel, estrai il colore RGB e convertilo in un indice di classe
        for y in range(mask_np.shape[0]):
            for x in range(mask_np.shape[1]):
                pixel_color = tuple(mask_np[y, x])
                # Usa il colore più vicino se il colore esatto non è presente
                if pixel_color in COLOR_LABEL_MAP:
                    label_mask[y, x] = COLOR_LABEL_MAP[pixel_color]
        
        
        # (520, 520, 1) -> (520, 520)
        label_mask = np.squeeze(label_mask)

        return label_mask
    
    class_indices = tf.py_function(
        func=convert_rgb_to_class_indices,
        inp=[mask],
        Tout=tf.int32
    )
    
    class_indices.set_shape([HEIGHT, WIDTH]) # (520, 520)

    return class_indices

def load() -> tf.data.Dataset :
    
    # base dir for imgs and masks
    imgs_dir = os.path.join(PATH, 'JPEGImages')
    masks_dir = os.path.join(PATH, 'SegmentationClass')

    # list of IDs fot the images of the val set
    split_file = os.path.join(PATH, 'ImageSets', 'Segmentation', "val.txt")
    with open(split_file, 'r') as f:
        image_ids = [line.strip() for line in f.readlines()]

    # list of images and masks paths of the val split
    data = []
    for img_id in image_ids:
        img_path = os.path.join(imgs_dir, f"{img_id}.jpg")
        mask_path = os.path.join(masks_dir, f"{img_id}.png")
    
        if os.path.exists(img_path) and os.path.exists(mask_path):
                data.append((img_path, mask_path))

    # create a tf.data.Dataset from the image and mask paths
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(
        lambda paths_tupple: (
            load_preprocess_image(paths_tupple[0]), 
            load_preprocess_mask(paths_tupple[1])
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # TODO : remove, just for debugging
    # 4 batches of 32, takes longer with the whole dataset
    dataset = dataset.take(6)


    return dataset