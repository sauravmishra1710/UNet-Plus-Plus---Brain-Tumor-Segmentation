import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, concatenate, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

class UNetPlusPlus():
    
    def __init__(self):
        pass
    
    def BuildNetwork(self, input_shape = (512, 512, 1), nb_classes = 1, deep_supervision = False):

        """
        Creates the UNet++ Netwrork for biomedical image segmentation.

        Args:
            input_shape (tuple): A shape tuple (integers), not including the batch size.
                                 Default value is (512, 512, 1).
            nb_classes (Integer): the dimensionality (no. of filters) of the output space 
                        (i.e. the number of output filters in the convolution).

            deep_supervision (boolean): A flag that toggles between the 2 different training modes -
                                        1) the ACCURATE mode - where the outputs from all segmentation 
                                           branches are averaged., 
                                        2) the FAST mode - wherein the final segmentation map is selected from 
                                           only one of the segmentation branches.
                                        Default vaue - False
        Returns:
            model: the neural network model representing the UNet++ model architechture.

        Reference: 
            UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
            https://arxiv.org/abs/1807.10165

        """
        filters = [64, 128, 256, 512, 1024]

        img_input = Input(shape = input_shape, name = 'InputLayer')

        conv00 = self.InsertConvolutionBlock(img_input, block_level='00', filters=filters[0])
        pool0 = MaxPooling2D(pool_size = 2, strides = 2, name='pool0')(conv00)

        conv10 = self.InsertConvolutionBlock(pool0, block_level='10', filters=filters[1])
        pool1 = MaxPooling2D(pool_size = 2, strides = 2, name='pool1')(conv10)

        up01 = Conv2DTranspose(filters = filters[0], kernel_size = 2, strides = 2, padding='same', name='upsample01')(conv10)
        conv01 = concatenate([up01, conv00], name='concat01')
        conv01 = self.InsertConvolutionBlock(conv01, block_level='01', filters=filters[0])

        conv20 = self.InsertConvolutionBlock(pool1, block_level='20', filters=filters[2])
        pool2 = MaxPooling2D(pool_size = 2, strides = 2, name='pool2')(conv20)

        up11 = Conv2DTranspose(filters = filters[1], kernel_size = 2, strides = 2, padding='same', name='upsample11')(conv20)
        conv11 = concatenate([up11, conv10], name='concat11')
        conv11 = self.InsertSkipPathway(conv11, block_level='11', filters=filters[1])

        up02 = Conv2DTranspose(filters = filters[0], kernel_size = 2, strides = 2, padding='same', name='upsample02')(conv11)
        conv02 = concatenate([up02, conv00, conv01], name='concat02')
        conv02 = self.InsertConvolutionBlock(conv02, block_level='02', filters=filters[0])

        conv30 = self.InsertConvolutionBlock(pool2, block_level='30', filters=filters[3])
        pool3 = MaxPooling2D(pool_size = 2, strides = 2, name='pool3')(conv30)

        up21 = Conv2DTranspose(filters = filters[2], kernel_size = 2, strides = 2, padding='same', name='upsample21')(conv30)
        conv21 = concatenate([up21, conv20], name='concat21')

        conv21 = self.InsertConvolutionBlock(conv21, block_level='21', filters=filters[2])

        up12 = Conv2DTranspose(filters = filters[1], kernel_size = 2, strides = 2, padding='same', name='upsample12')(conv21)
        conv12 = concatenate([up12, conv10, conv11], name='concat12')
        conv12 = self.InsertSkipPathway(conv12, block_level='12', filters=filters[1])

        up03 = Conv2DTranspose(filters = filters[0], kernel_size = 2, strides = 2, padding='same', name='upsample03')(conv12)
        conv03 = concatenate([up03, conv00, conv01, conv02], name='concat03')
        conv03 = self.InsertConvolutionBlock(conv03, block_level='03', filters=filters[0])

        conv40 = self.InsertConvolutionBlock(pool3, block_level='40', filters=filters[4])

        up31 = Conv2DTranspose(filters = filters[3], kernel_size = 2, strides = 2, padding='same', name='upsample31')(conv40)
        conv31 = concatenate([up31, conv30], name='concat31')
        conv31 = self.InsertSkipPathway(conv31, block_level='31', filters=filters[3])

        up22 = Conv2DTranspose(filters = filters[2], kernel_size = 2, strides = 2, padding='same', name='upsample22')(conv31)
        conv22 = concatenate([up22, conv20, conv21], name='concat22')
        conv22 = self.InsertSkipPathway(conv22, block_level='22', filters=filters[2])

        up13 = Conv2DTranspose(filters = filters[1], kernel_size = 2, strides = 2, padding='same', name='upsample13')(conv22)
        conv13 = concatenate([up13, conv10, conv11, conv12], name='concat13')
        conv13 = self.InsertSkipPathway(conv13, block_level='13', filters=filters[1])

        up04 = Conv2DTranspose(filters = filters[0], kernel_size = 2, strides = 2, padding='same', name='upsample04')(conv13)
        conv04 = concatenate([up04, conv00, conv01, conv02, conv03], name='concat04')
        conv04 = self.InsertConvolutionBlock(conv04, block_level='04', filters=filters[0])

        nested_op_1 = Conv2D(nb_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding='same', name='op1')(conv01)

        nested_op_2 = Conv2D(nb_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding='same', name='op2')(conv02)

        nested_op_3 = Conv2D(nb_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding='same', name='op3')(conv03)

        nested_op_4 = Conv2D(nb_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding='same', name='op4')(conv04)

        if deep_supervision:
            output = [nested_op_1, nested_op_2, nested_op_3, nested_op_4]
        else:
            output = [nested_op_4]

        model = Model(inputs = img_input, outputs = output, name = "UNet++")

        return model

    def InsertSkipPathway(self, input_tensor, block_level, filters, kernel_size = 3):

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

        Reference: 
            UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
            https://arxiv.org/abs/1807.10165

        """
        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_1')(input_tensor)

        x = Dropout(rate = 0.5, name = 'dp' + block_level + '_1')(x)

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_2')(x)

        x = Dropout(rate = 0.5, name = 'dp' + block_level + '_2')(x)

        return x

    def InsertConvolutionBlock(self, input_tensor, block_level, filters, kernel_size = 3):

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

        Reference: 
            UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
            https://arxiv.org/abs/1807.10165

        """

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_1')(input_tensor)

        x = Dropout(rate = 0.5, name = 'dp' + block_level + '_1')(x)

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_2')(x)

        x = Dropout(rate = 0.5, name = 'dp' + block_level + '_2')(x)

        return x