import numpy as np
import tensorflow as tf 

keras = tf.keras

class VOCDataset:
    '''
    Dataset of VOC
    Args:
        root_dir: str, root dir of the voc dev kit
        which_split: str, one from 'train', 'val', 'test'
        classes: list, list of classes to be trained
    '''
    # PASCAL VOC 21 Classes name and their colors
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
    def __repr__(self):
        return "PascalVOCDataset"
    
    def __init__(
        self,
    ):
        pass
        
    def __len__(self):
        return 10
    
    def __getitem__(self,index):

        img = tf.ones((520, 520, 3))
        mask = tf.ones((520, 520))
        mask = tf.cast(mask, tf.uint8)

        return img, mask

    def _encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.PASCAL_COLORS):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        
        return label_mask

#dataloader class
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = tf.range(len(dataset))

        self.on_epoch_end()

    def batch(self, new_number):
        self.batch_size = new_number
        return self

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [tf.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch

    def __repr__(self):
        return self.dataset.__repr__()
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
