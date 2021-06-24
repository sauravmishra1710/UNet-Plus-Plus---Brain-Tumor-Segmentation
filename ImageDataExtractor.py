import os
import h5py
import glob
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm

class ImageDataExtractor():
    
    
    def __init__(self, mat_data_path, img_data_path, mask_data_path):
        
        self.MAT_DATA_PATH = mat_data_path
        self.IMG_DATA_PATH = img_data_path
        self.MASK_DATA_PATH = mask_data_path
    
    def __readMatData(self, filePath: str):
    
        """ 
        Reads the mat file and returns the image & mask array.

        Args:
            filePath(str): Path of the file to be read.

        Returns:
            data(dict): The array of the image and the corresponding mask 
                        in the dictionary format.
                        'image': The numpy array for image.
                        'mask' : The numpy array for the corresponding mask.

        """

        file = h5py.File(filePath, 'r')

        imgData = dict()
        imgData['image'] = np.array(file.get('cjdata/image'))
        imgData['mask'] = np.array(file.get('cjdata/tumorMask'))

        return imgData

    def __create_dir(self, target_dir):

        """
        Creates folder if there is no folder in the 
        specified directory path.

        Args: 
            target_folder(str): path of the folder which needs to be created.

        Returns: 
            None

        """
        if not (os.path.isdir(target_dir)):
            os.mkdir(target_dir)

    def __save_image_data(self, filename, data, imgFormat = 'png'):

        """ 
        Saves the image & mask array in png format.

        Args:
            filename(str): Name of the file without the extension.
            data(dict): The array of the image and the corresponding mask 
                        in the dictionary format.
                        'image': The numpy array for image.
                        'mask' : The numpy array for the corresponding mask.

        Returns: 
            None

        """

        img_path = os.path.join(self.IMG_DATA_PATH, filename + '.' + imgFormat)
        mask_path = os.path.join(self.MASK_DATA_PATH, filename + '.' + imgFormat)
        
        mpimg.imsave(img_path, data['image'], cmap = 'gray', format = imgFormat)
        mpimg.imsave(mask_path, data['mask'], cmap = 'gray', format = imgFormat)
        
    def extractAndSaveImages(self):
        
        """ 
        Extracts the image data from the corresponding .mat files and
        saves the extracted image & mask array in png format.

        Args:
            None

        Returns: 
            None

        """
        
        self.__create_dir(self.IMG_DATA_PATH)
        self.__create_dir(self.MASK_DATA_PATH)
        
        files = glob.glob(self.MAT_DATA_PATH + '\*.mat')
        
        for idx in  tqdm(range(1, len(files) + 1)):
            file = files[idx]
            
            # extract the filename
            filename = os.path.splitext(os.path.basename(file))[0]
            data = self.__readMatData(file)
            self.__save_image_data(filename, data)