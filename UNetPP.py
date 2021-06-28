import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, concatenate, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

class UNetPlusPlus():
    
    """ 
    Unet++ Model design.
    
    This module consumes the Unet utilities framework moule and designs the Unet network.
    It consists of a contracting path and an expansive path. Both these paths are joined 
    by a bottleneck block.
    
    The different blocks involved in the design of the network can be referenced @ 
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    
    Reference:
        [1] UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
            https://arxiv.org/abs/1807.10165
            
        [2] https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical
    """
    
    def __init__(self, input_shape = (512, 512, 1), filters = [32, 64, 128, 256, 512], nb_classes = 1, deep_supervision = False):
        
        """
        Initialize the Unet framework and the model parameters - input_shape, 
        filters and padding type. 
        
        Args:
            input_shape (tuple): A shape tuple (integers), not including the batch size.
                                 Default value is (512, 512, 1).
                                 
            filters (array of integers: a collection of filters denoting the number of components to be used at each blocks along the 
                        contracting and expansive paths. The original paper implementation for number of filters along the 
                        contracting and expansive paths are [32, 64, 128, 256, 512]. (as per paper: k = 32 × 2^i).
                        
            nb_classes (Integer): the dimensionality (no. of filters) of the output space .
                        (i.e. the number of output filters in the convolution).

            deep_supervision (boolean): A flag that toggles between the 2 different training modes -
                                        1) the ACCURATE mode - where the outputs from all segmentation 
                                           branches are averaged., 
                                        2) the FAST mode - wherein the final segmentation map is selected from 
                                           only one of the segmentation branches.
                                        Default vaue - False
            
        **Remarks: The default values are as per the implementation in the original paper @ https://arxiv.org/pdf/1505.04597
         
        """

        self.__input_shape = input_shape
        self.__filters = filters
        self.__nb_classes = nb_classes
        self.__deep_supervision = deep_supervision
    
    def BuildNetwork(self):

        """
        Creates the UNet++ Netwrork for biomedical image segmentation.

        Args:
            None
            
        Returns:
            model: the neural network model representing the UNet++ model architechture.

        """

        input_img = Input(shape = self.__input_shape, name = 'InputLayer')

        conv00 = self.__InsertConvolutionBlock(input_img, block_level = '00', filters = self.__filters[0])
        pool0 = MaxPooling2D(pool_size = 2, strides = 2, name = 'pool0')(conv00)

        conv10 = self.__InsertConvolutionBlock(pool0, block_level = '10', filters = self.__filters[1])
        pool1 = MaxPooling2D(pool_size = 2, strides = 2, name = 'pool1')(conv10)

        up01 = Conv2DTranspose(filters = self.__filters[0], kernel_size = 2, strides = 2, padding='same', name='upsample01')(conv10)
        conv01 = concatenate([up01, conv00], name='concat01')
        conv01 = self.__InsertConvolutionBlock(conv01, block_level = '01', filters = self.__filters[0])

        conv20 = self.__InsertConvolutionBlock(pool1, block_level = '20', filters = self.__filters[2])
        pool2 = MaxPooling2D(pool_size = 2, strides = 2, name = 'pool2')(conv20)

        up11 = Conv2DTranspose(filters = self.__filters[1], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample11')(conv20)
        conv11 = concatenate([up11, conv10], name = 'concat11')
        conv11 = self.__InsertSkipPathway(conv11, block_level = '11', filters = self.__filters[1])

        up02 = Conv2DTranspose(filters = self.__filters[0], kernel_size = 2, strides = 2, padding='same', name='upsample02')(conv11)
        conv02 = concatenate([up02, conv00, conv01], name = 'concat02')
        conv02 = self.__InsertConvolutionBlock(conv02, block_level = '02', filters = self.__filters[0])

        conv30 = self.__InsertConvolutionBlock(pool2, block_level = '30', filters = self.__filters[3])
        pool3 = MaxPooling2D(pool_size = 2, strides = 2, name = 'pool3')(conv30)

        up21 = Conv2DTranspose(filters = self.__filters[2], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample21')(conv30)
        conv21 = concatenate([up21, conv20], name='concat21')

        conv21 = self.__InsertConvolutionBlock(conv21, block_level='21', filters = self.__filters[2])

        up12 = Conv2DTranspose(filters = self.__filters[1], kernel_size = 2, strides = 2, padding='same', name = 'upsample12')(conv21)
        conv12 = concatenate([up12, conv10, conv11], name = 'concat12')
        conv12 = self.__InsertSkipPathway(conv12, block_level = '12', filters = self.__filters[1])

        up03 = Conv2DTranspose(filters = self.__filters[0], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample03')(conv12)
        conv03 = concatenate([up03, conv00, conv01, conv02], name = 'concat03')
        conv03 = self.__InsertConvolutionBlock(conv03, block_level = '03', filters = self.__filters[0])

        conv40 = self.__InsertConvolutionBlock(pool3, block_level = '40', filters = self.__filters[4])

        up31 = Conv2DTranspose(filters = self.__filters[3], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample31')(conv40)
        conv31 = concatenate([up31, conv30], name = 'concat31')
        conv31 = self.__InsertSkipPathway(conv31, block_level = '31', filters=self.__filters[3])

        up22 = Conv2DTranspose(filters = self.__filters[2], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample22')(conv31)
        conv22 = concatenate([up22, conv20, conv21], name = 'concat22')
        conv22 = self.__InsertSkipPathway(conv22, block_level = '22', filters = self.__filters[2])

        up13 = Conv2DTranspose(filters = self.__filters[1], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample13')(conv22)
        conv13 = concatenate([up13, conv10, conv11, conv12], name='concat13')
        conv13 = self.__InsertSkipPathway(conv13, block_level = '13', filters = self.__filters[1])

        up04 = Conv2DTranspose(filters = self.__filters[0], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample04')(conv13)
        conv04 = concatenate([up04, conv00, conv01, conv02, conv03], name = 'concat04')
        conv04 = self.__InsertConvolutionBlock(conv04, block_level = '04', filters = self.__filters[0])

        nested_op_1 = Conv2D(filters = self.__nb_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding = 'same', name = 'op1')(conv01)

        nested_op_2 = Conv2D(filters = self.__nb_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding = 'same', name = 'op2')(conv02)

        nested_op_3 = Conv2D(filters = self.__nb_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding= 'same', name = 'op3')(conv03)

        nested_op_4 = Conv2D(filters = self.__nb_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding = 'same', name = 'op4')(conv04)

        if self.__deep_supervision:
            output = [nested_op_1, nested_op_2, nested_op_3, nested_op_4]
        else:
            output = [nested_op_4]

        model = Model(inputs = input_img, outputs = output, name = "UNet++")

        return model

    def __InsertSkipPathway(self, input_tensor, block_level, filters, kernel_size = 3):

        """
        Inserts a convolution block along the skip pathways.

        Args:
            input_tensor: The input that would go into the convolutional block.
            block_level: the level of the current block.
            filters: the dimensionality (no. of filters) of the output space 
                        (i.e. the number of output filters in the convolution).
            kernel_size: the size of the convolving window. Default value is 3.
                         All convolutional layers along a skip pathway (X(i, j) )
                         use k kernels of size 3×3.

        Returns:
            x: The 2D convolution output.

        """
        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_1')(input_tensor)

        x = Dropout(rate = 0.5, name = 'dp' + block_level + '_1')(x)

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_2')(x)

        x = Dropout(rate = 0.5, name = 'dp' + block_level + '_2')(x)

        return x

    def __InsertConvolutionBlock(self, input_tensor, block_level, filters, kernel_size = 3):

        """
        Inserts a convolution block along the contracting 
        and expanding paths of the network.

        Args:
            input_tensor: The input that would go into the convolutional block.
            block_level: the level of the current block.
            filters: the dimensionality (no. of filters) of the output space 
                        (i.e. the number of output filters in the convolution).
            kernel_size: the size of the convolving window. Default value is 3.            

        Returns:
            x: The 2D convolution output.

        """

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_1')(input_tensor)

        x = Dropout(rate = 0.5, name = 'dp' + block_level + '_1')(x)

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_2')(x)

        x = Dropout(rate = 0.5, name = 'dp' + block_level + '_2')(x)

        return x
    
    def CompileAndSummarizeModel(self, model, optimizer = "adam", loss = "binary_crossentropy"):

        """
        Compiles and displays the model summary of the Unet++ model.

        Args:
            model: The keras instance of the Unet++ model.
            optimizer: model optimizer. Default is the adam optimizer.
            loss: the loss function. Default is the binary cross entropy loss.

        Return:
            None

        """
        model.compile(optimizer = optimizer, loss = loss, metrics = ["acc"])
        model.summary()

    def plotModel(self, model, to_file = 'unetpp.png', show_shapes = True, dpi = 96):

        """
        Saves the Unet++ model plot/figure to a file.

        Args:
            model: The keras instance of the Unet++ model.
            to_file: the file name to save the model. Default name - 'unet.png'.
            show_shapes: whether to display shape information. Default = True.
            dpi: dots per inch. Default value is 96.

        Return:
            None

        """

        tf.keras.utils.plot_model(model, to_file = to_file, show_shapes = show_shapes, dpi = dpi)