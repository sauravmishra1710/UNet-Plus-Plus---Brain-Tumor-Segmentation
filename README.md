# Brain-Tumor-Segmentation

Brain tumor segmentation using UNet++ Architecture 

## UNet++ Introduction
UNet++, a convolutional neural network dedicated for biomedical image segmentation, was designed, and applied in 2018 by (Zhou et al., 2018). UNet++ was basically designed to overcome some of the short comings of the UNet architecture. UNet works on the idea of skip connections. U-Net concatenates them and add convolutions and non-linearities between each up-sampling block. The skip connections recover the full spatial resolution at the network output, making fully convolutional methods suitable for semantic segmentation. UNet and other segmentation models based on the encoder-decoder architecture tend to fuse semantically dissimilar feature maps from the encoder and decoder sub-networks, which may degrade segmentation performance. This is where UNet++ is shown to have an edge over the other players as it bridges the semantic gap between the feature maps of the encoder and decoder prior to fusion thus improving the segmentation performance and output.

## Evolution of UNet++
UNet and FCNs have attained the state-of-the-art status in the field of medical image segmentation. The encoder-decoder structure are widely used in almost every semantic and instance segmentation task. Their success is largely attributed to the design of the skip connections that combine the deep, semantic, coarse-grained feature maps from the decoder sub-network with shallow, low-level, fine-grained feature maps from the encoder sub-network. However, the network structure and the design of the skip connections suffer from the following limitations. 
1.	The network depth could vary from task to task largely attributed to the amount of the data available and the complexity of the segmentation task.
2.	The design of skip connections employed is very much restrictive, such that it expects encoder and decoder feature maps to be fused be at the same scale.

The evolution goes through 3 different architectural phases with each phase improving the limitations of the previous one. The three different phases are  - 
1.	Ensemble UNets
2.	UNet+
3.	UNet++

## Architecture
***“UNet++ is constructed from U-Nets by connecting the decoders, resulting in densely connected skip connections, enabling dense feature propagation along skip connections and thus more flexible feature fusion at the decoder nodes. As a result, each node in the UNet++ decoders, from a horizontal perspective, combines multiscale features from its all preceding nodes at the same resolution, and from a vertical perspective, integrates multiscale features across different resolutions from its preceding node. This multiscale feature aggregation of UNet++ gradually synthesizes the segmentation, leading to increased accuracy and faster convergence.”***

![UNet++ Architecture](https://github.com/sauravmishra1710/UNet-Plus-Plus---Brain-Tumor-Segmentation/blob/main/images/nested_unet_architecture.PNG)

## Model Training and Results
UNet++ model is designed and put to training for a brain tumor segmentation task. The network parameters are chosen as per the implementation in the original paper (Zhou et al., 2018). The model is trained over 30 epochs on brain tumor data available at (Brain Tumor Dataset, n.d.). The dataset consists of 3064 brain tumor images along with their masks. For training purpose, the data is divided into training, validation and tests sets each having 2800, 200 and 64 images respectively.

### Training Parameters
The network is trained with the below parameters set.
**Epochs**: 30
**Batch Size**: 64
**Image Size**: (128, 128)
**Loss Function and Metric**: Combination of Binary Cross Entropy and Dice Coefficient, IoU coefficient. 
For this task, early stopping was not considered.
**Dataset Used**: Brain Tumor dataset @ https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

### Implementation Notebooks

1. UNet++ Design and Implementation -
    - Notebook @ https://github.com/sauravmishra1710/UNet-Plus-Plus---Brain-Tumor-Segmentation/blob/main/UNetPlusPlus%20-%20Nested%20UNet.ipynb
    - Utility Files @ https://github.com/sauravmishra1710/UNet-Plus-Plus---Brain-Tumor-Segmentation/blob/main/UNetPP.py

2. Brain Tumor Segmentation - 
    - Notebook @ https://github.com/sauravmishra1710/UNet-Plus-Plus---Brain-Tumor-Segmentation/blob/main/BrainTumorSegmentation.ipynb

3. Documentation @ https://github.com/sauravmishra1710/UNet-Plus-Plus---Brain-Tumor-Segmentation/blob/main/UNetPlusPlus.pdf

### Segmentation Results
The image below shows the segmentation results from some of the images from the test set.

![Brain Tumor Segmentation Results](https://github.com/sauravmishra1710/UNet-Plus-Plus---Brain-Tumor-Segmentation/blob/main/images/Segmentation_Results.png)

Comparing the original image, original mask and the predicted mask, the model based on the UNet++ architecture is correctly able to segment the brain tumor location and generate the masks. Though there are some differences seen in the visualizations above, these can be improved with further training and fine tuning the model itself.

## Summary

•	UNet++ aims to improve segmentation accuracy, with a series of nested, dense skip pathways.

•	Redesigned skip pathways make optimization easier by getting the semantically similar feature maps.

•	Dense skip connections improve segmentation accuracy and make the gradient flow smoother.

•	Deep supervision allows for model complexity tuning to balance between speed and performance optimization by allowing the model to toggle between 2 different training modes in the fast mode and the accurate mode.

•	UNet++ differs from the original U-Net in three ways - (refer architecture diagram above)

  - It has convolution layers (green)on skip pathways, which bridges the semantic gap between encoder and decoder feature maps.
    
  - It has dense skip connections on skip pathways (blue), which improves gradient flow.
    
  - It employs deep supervision (red), which enables model pruning and improves or in the worst case achieves comparable performance to using only one loss layer.

## References

1.	brain tumor dataset. (n.d.). Retrieved July 5, 2021, from https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 <br>
2.	Turečková, A., Tureček, T., Komínková Oplatková, Z., & Rodríguez-Sánchez, A. (2020). Improving CT Image Tumor Segmentation Through Deep Supervision and Attentional Gates. Frontiers in Robotics and AI, 7, 106. https://doi.org/10.3389/frobt.2020.00106 <br>
3.	UNet++: A Nested U-Net Architecture for Medical Image Segmentation | Papers With Code. (n.d.). Retrieved July 5, 2021, from https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical <br>
4.	Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. https://arxiv.org/abs/1807.10165 <br>
5.	Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2019). UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation. http://arxiv.org/abs/1912.05074


