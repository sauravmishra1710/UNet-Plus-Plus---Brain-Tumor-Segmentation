import os
import stat
import h5py
import glob
import shutil
import zipfile
import requests
import numpy as np
from pathlib import Path
from tqdm import notebook
import matplotlib.image as mpimg

class ImageDataExtractor():
    
    """
    A utility class to download, organize and read the *.mat files that are saved in the matlab format, 
    extract the image data that is stored as part of the file.
    The image and the corresponding tumor mask are stored as part of the fields 
    cjdata.image & cjdata.tumorMask.
    
    """
    
    def __init__(self):
        
        self.__DATA = 'data'
        self.__MAT_DATA_PATH = 'data\\matData'
        self.__IMG_DATA_PATH = 'data\\imgData\\img'
        self.__MASK_DATA_PATH = 'data\\imgData\\mask'
        self.__TEMP_DOWNLOAD_PATH = os.path.join(self.__DATA, 'temp\\download')
        self.__ZIP_FILE = os.path.join(self.__TEMP_DOWNLOAD_PATH, '1512427.zip')
        self.__TEMP_ZIP_FILE = os.path.join(os.path.split(self.__ZIP_FILE)[0], 
                                            os.path.splitext(os.path.basename(self.__ZIP_FILE))[0] + '__.zip')
        self.__TEMP_UNZIP_PATH = os.path.join(Path(self.__TEMP_DOWNLOAD_PATH).parent.name, 'unzip')
        self.__DATA_URL = 'https://ndownloader.figshare.com/articles/1512427/versions/5'
        self.README_PATH = os.path.join(self.__DATA, 'README.txt')
        
        # if the master 'data' folder is not present
        # create the directory and proceed.
        if not (os.path.isdir(self.__DATA)):
            self.__create_dir(self.__DATA)
        
        # if the temp download folder is not present, create it
        if not (os.path.isdir(self.__TEMP_DOWNLOAD_PATH)):
            self.__create_dir(self.__TEMP_DOWNLOAD_PATH)
        
        # if the temp data unzip folder is not present, create it
        if not (os.path.isdir(self.__TEMP_UNZIP_PATH)):
            self.__create_dir(self.__TEMP_UNZIP_PATH)
        
    
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
        
        # the image and the corresponding mask are stored as part of the fields - 
        # cjdata.image & cjdata.tumorMask
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
        
        # create the directory/folder if it is not already 
        # present in the specified path.
        if not (os.path.isdir(target_dir)):
            os.makedirs(target_dir, exist_ok = True)

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

        img_path = os.path.join(self.__IMG_DATA_PATH, filename + '.' + imgFormat)
        mask_path = os.path.join(self.__MASK_DATA_PATH, filename + '.' + imgFormat)
        
        mpimg.imsave(img_path, data['image'], cmap = 'gray', format = imgFormat)
        mpimg.imsave(mask_path, data['mask'], cmap = 'gray', format = imgFormat)
        
    def downloadAndExtractImages(self):
        
        """ 
        Extracts the image data from the corresponding .mat files and
        saves the extracted image & mask array in png format.

        Args:
            None

        Returns: 
            None

        """ 
        
        # check if the data is already extracted. check the relevant directories are created or not.
        if os.path.isdir(self.__MAT_DATA_PATH) and os.path.isdir(self.__IMG_DATA_PATH) and os.path.isdir(self.__MASK_DATA_PATH):
            print(">>> Data already downloaded. Check the following directoies - ")
            print(">>> Mat files located @ " + "'" + self.__MAT_DATA_PATH + "'")
            print(">>> Image files located @ " + "'" + self.__IMG_DATA_PATH + "'")
            print(">>> Mask files located @ " + "'" + self.__MASK_DATA_PATH + "'")
            return
        
        # download & unzip the data if not present.
        if not (os.path.isdir(self.__MAT_DATA_PATH)):
            self.__downloadData()
            self.__upzipData()
        
        # extract the image amd the corresponding mask data.
        if not (os.path.isdir(self.__IMG_DATA_PATH) and os.path.isdir(self.__MASK_DATA_PATH)):
            # create the directory/folder if it is not already 
            # present in the specified path.
            self.__create_dir(self.__IMG_DATA_PATH)
            self.__create_dir(self.__MASK_DATA_PATH)

            # extract the .mat files into a list.
            files = glob.glob(self.__MAT_DATA_PATH + '\*.mat')

            print(">>> Extracting images and masks...")

            for idx in  notebook.tqdm(range(len(files))):

                file = files[idx]

                # extract the filename to be used to save the 
                # image and its mask.
                filename = os.path.splitext(os.path.basename(file))[0]

                data = self.__readMatData(file)
                self.__save_image_data(filename, data)

            print(">>> Data extraction complete...")
        
            print(">>> Removing the master zip file...")
            if os.path.isfile(self.__ZIP_FILE):
                os.remove(self.__ZIP_FILE)
            
    def __downloadData(self, chunk_size = 1024):
    
        """ 
        Download the file from the given url.

        Args:
            chunk_size (int):  number of bytes it should read into memory. Default Value is 1024

        Returns: 
            None

        """

        # Delete the incomplete downloads from previous sessions.
        if os.path.isfile(self.__TEMP_ZIP_FILE):
            print('>>> Deleting any incomplete downloaded file from previous session @ ' + self.__TEMP_ZIP_FILE)
            os.remove(self.__TEMP_ZIP_FILE)

        # Download the file
        print(">>> Downloading data to - " + "'" + self.__TEMP_ZIP_FILE + "'")
        response = requests.get(self.__DATA_URL, stream = True)
        with open(self.__TEMP_ZIP_FILE, "wb") as handle:

            total_size = round(int(response.headers['Content-Length']), 3)
            pbar = notebook.tqdm(unit = "B", total = total_size)
            for chunk in response.iter_content(chunk_size = chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    handle.write(chunk)
                    pbar.update(len(chunk))

        # Rename the file to the correct name 
        # once download is complete.
        os.rename(self.__TEMP_ZIP_FILE, self.__ZIP_FILE)
        print(">>> Download Complete...")
        
    def __upzipData(self):
        
        """
        extracts the downloaded data, data readme file
        and prepares to read the images and masks.
        
        Args:
            None
        
        Return:
            None
        
        """
    
        self.__create_dir(self.__TEMP_UNZIP_PATH)

        # extract the master zipped file.
        print(">>> Extracting Master Folder...")
        for idx in  notebook.tqdm(range(1)):
            with zipfile.ZipFile(self.__ZIP_FILE, "r") as _zip:
                _zip.extractall(self.__TEMP_UNZIP_PATH)

        print(">>> Extracting *.mat files...")
        files = glob.glob(self.__TEMP_UNZIP_PATH + '\*.zip')

        self.__create_dir(self.__MAT_DATA_PATH)

        # exract the mat files from the respective zipped files.
        for idx in  notebook.tqdm(range(len(files))):
                with zipfile.ZipFile(files[idx], "r") as _zip:
                    _zip.extractall(self.__MAT_DATA_PATH)
        
        # copy the data readme file to the data folder.
        readMeDestination = os.path.split(self.__MAT_DATA_PATH)[0]
        readMeFileName = os.path.split('data\\temp\\unzip\\README.txt')[1]
        print(">>> Copying the data ReadMe file to - " + "'\\" + os.path.join(readMeDestination, readMeFileName) + "'")
        readMe = glob.glob(self.__TEMP_UNZIP_PATH + '\*.txt')
        shutil.copy2(readMe[0], readMeDestination)
        
        print(">>> Data unzipped successfully to " + "'" + self.__MAT_DATA_PATH + "'")
        # delete the temp folder @ temp_unzip_path
        if (os.path.isdir(self.__TEMP_UNZIP_PATH)):
            print(">>> Removing the temp download folder..." + "'" + self.__TEMP_DOWNLOAD_PATH + "'")
            shutil.rmtree(self.__TEMP_DOWNLOAD_PATH)
            