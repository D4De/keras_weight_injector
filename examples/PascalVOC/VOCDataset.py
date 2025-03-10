import numpy as np
import tensorflow as tf 
import os
import PIL
import math

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
        root_dir,
        which_split,
        classes=None,
        transform=None,
        target_transform = None,
        HEIGHT = 640, WIDTH = 640,
    ):
        #convert chosen class name to class index
        self.class_values = [self.PASCAL_CLASSES.index(cls.lower()) for cls in classes]
        #root dir
        self.root_dir = root_dir
        #.jpg and .png file dir
        imgs_dir = os.path.join(root_dir,'JPEGImages')
        masks_dir = os.path.join(root_dir,'SegmentationClass')
        #image name list .txt file dir 
        masktxt_dir = os.path.join(root_dir,'ImageSets','Segmentation')
        classtxt_dir = os.path.join(root_dir,'ImageSets','Main')
        #create the file id list of all images with masks
        file_path = os.path.join(masktxt_dir, which_split + ".txt")
        file_list = tuple(open(file_path, "r"))
        mask_ids = [id_.rstrip() for id_ in file_list]
        
        #if only one class is chosen, create the file id list of this explict chosen class
        if len(classes)==1:
            c = classes[0]
            file_path= pjoin(classtxt_dir,c+'_'+which_split+'.txt')
            file_list = tuple(open(file_path, "r"))
            #slice op here: class string ends with '1' if exists in the image, otherwise '-1'
            class_ids = [id_[:6] for id_ in file_list if id_[7] != '-']
            self.ids = [id_ for id_ in class_ids if id_ in mask_ids]
        else:
            self.ids = mask_ids
            
        #create filepath
        self.img_filepaths = [os.path.join(imgs_dir, img_id)+'.jpg' for img_id in self.ids]
        self.mask_filepaths = [os.path.join(masks_dir,img_id)+'.png' for img_id in self.ids]

        self.transform = transform
        self.target_transform = target_transform

    def __repr__(self):
        return "PascalVOCDataset"
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,index):
        img_name = self.img_filepaths[index]
        mask_name = self.mask_filepaths[index]
        
        # Load all images in RAM
        img = PIL.Image.open(img_name)
        mask = PIL.Image.open(mask_name)

        img = tf.convert_to_tensor(img)
        mask = tf.convert_to_tensor(mask)
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

         
            """img = img.resize((HEIGHT,WIDTH))
            img = np.array(img)
            images.append(img) """
        
        #extracts chosen classes as one-hot code
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        
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

    def batch(self, new_number):
        self.batch_size = new_number
        return self

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            try:
                data.append(self.dataset[j])
            except IndexError:
                break
        
        # transpose list of lists
        batch = [tf.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch

    def __repr__(self):
        return self.dataset.__repr__()
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.indexes) / self.batch_size)
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
