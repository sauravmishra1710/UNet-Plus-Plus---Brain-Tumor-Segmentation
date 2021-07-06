import os
import sys
import random
import numpy as np
import cv2
import gc
import tensorflow as tf

class ImageDataGen(tf.keras.utils.Sequence):
    
    """
    The custom data generator class generates and feeds data to
    the model dynamically in batches during the training phase.
    
    This generator generates batched of data for the dataset available @
    Find the nuclei in divergent images to advance medical discovery -
    https://www.kaggle.com/c/data-science-bowl-2018
    
    **
    tf.keras.utils.Sequence is the root class for 
    Custom Data Generators.
    **
    
    Args:
        image_ids: the ids of the image.
        img_path: the full path of the image directory.
        mask_path: the full path of the mask directory.
        batch_size: no. of images to be included in a batch feed. Default is set to 32.
        image_size: size of the image. Default is set to 512 as per the data available.
        
    Ref: https://dzlab.github.io/dltips/en/keras/data-generator/
    
    """
    def __init__(self, image_ids, img_path, mask_path, batch_size = 32, image_size = 512, shuffle = True):
        
        self.ids = image_ids
        self.img_path = img_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __load__(self, item):
        
        """
        loads the specified image.
        
        """
        
        full_image_path = os.path.join(self.img_path, item)
        full_mask_path = os.path.join(self.mask_path, item)
        
        # load the images
        image = cv2.imread(full_image_path, 0)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # load the masks
        mask = cv2.imread(full_mask_path, 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # normalize the mask and the image. 
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        
        """
        Returns a single batch of data.
        
        Args:
            index: the batch index.
        
        """
        
        # edge case scenario where there are still some items left
        # after segregatings the images into batches of size batch_size.
        # the items left out will form one batch at the end.
        if(index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size
        
        # group the items into a batch.
        batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        
        image = []
        mask  = []
        
        # load the items in the current batch
        for item in batch:
            img, msk = self.__load__(item)
            image.append(img)
            mask.append(msk)
        
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        
        """
        optional method to run some logic at the end of each epoch: e.g. reshuffling
        
        """
        
        if self.shuffle:
            random.shuffle(self.ids)
            
        gc.collect()
    
    def __len__(self):
        
        """
        Returns the number of batches
        """
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
    