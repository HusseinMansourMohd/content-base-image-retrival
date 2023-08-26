# Content-Based Image Retrieval Using Deep Neural Networks

This project aims to improve the efficiency and precision of content-based image retrieval (CBIR) using deep neural networks. It is focused on overcoming traditional limitations associated with text-based image retrieval, low-level features-based approaches, and local features-based approaches. Specifically, it introduces the application of EfficientNets, R-MAC vectors, diffusion process, and fine-tuning to enhance CBIR performance. 


## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Implementation](#implementation)
4. [Results and Comparison](#results-and-comparison)
5. [Conclusion](#conclusion)

## Introduction

Text-based image retrieval approaches, such as manual annotation, lack a real understanding of image content. Other low-level feature-based approaches offer only partial representations, requiring extensive pre- and post-processing. Similarly, local features-based approaches are computationally expensive and not suitable for large-scale applications. This project introduces a CBIR system using EfficientNets, which offers an efficient and scalable solution.

## Methodology

1. **EfficientNets**: This model leverages depth-wise separable convolutions, compound model scaling, linear bottlenecks, and inverted residuals to improve the efficiency of image retrieval.

2. **R-MAC Vectors**: To overcome the limitations of traditional max/avg pooling, R-MAC vectors offer better object localization by encoding several image regions and performing simple aggregation methods.

3. **Diffusion Process**: This approach helps capture the data manifold in the feature space, which traditional ranking techniques often fail to do.

4. **Fine Tuning**: Fine-tuning is carried out to enhance the detection for specific data by freezing all layers except the final classification layer, replacing it with new layers and training them on top of the existing pre-trained network.

## Implementation

The retrieval pipeline follows these steps:
- Extract features from images using EfficientNet
- Apply R-MAC vectors to improve object localization
- Use a diffusion process to capture the manifold of the data
- Fine-tune the model to enhance specific detection

## Results and Comparison

The performance of the system was evaluated using the Mean Average Precision (mAP) on the Paris6k dataset. The project compares the results obtained with VGG-16, ResNet-50, and EfficientNet-B1, and the performance was enhanced significantly using R-MAC vectors and the diffusion process.

For the full comparison results, please refer to the presentation [here](https://github.com/HusseinMansourMohd/content-base-image-retrival/blob/master/presentation/CBIR%20Presentation.pdf).

## Conclusion

The project demonstrates that EfficientNets can significantly improve the performance of content-based image retrieval tasks. By understanding the retrieval pipeline and optimizing various factors, one can achieve more efficient and accurate image retrieval. 

Please feel free to raise any questions or issues [here](https://github.com/HusseinMansourMohd/content-base-image-retrival/issues).

## Acknowledgments

This project is a result of the collaborative effort of [Amel Jamal Yassin](https://github.com/ameljamal) and [Hussein Mansour Mohammed](https://github.com/HusseinMansourMohd). We greatly appreciate their contribution to the field.
