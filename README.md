# content-base-image-retrival
#Abstract
With the exploding number of images being generated every day, image retrieval has
quickly become an integral part of many modern day applications, which in turn puts
new constraints on the level of accepted accuracy for retrieval systems, this being
a subject that has been explored vigorously over the years, it is still considered to
be far from solved, as many systems fail to capture the semantic content of images
while meeting resource constraints.
In this work, a various number of techniques used for efficient image retrieval has
been investigated to highlight the breakthroughs in this field over the past years,
which all ultimately pointed at artificial intelligence & machine learning algorithms
as being the clear front runner in the race to create efficient content-based image
retrieval systems.
In particular the use of convolutional features of various neural networks was studied
and compared to give better insight into why these features represent optimal image
features for the image retrieval task .
The theory behind the newly emerging family of CNNs, known as Efficient-Nets was
investigated to understand the capabilities offered by the compound model scaling
approach, a number of variants of these networks were compared against each other
to determine the best suited iteration for the available resources, EfficientNet-B1
was selected and then benchmarked against popular image retrieval architectures;
in particular the VGG-16 and ResNet-50 were chosen, this has shown to achieve an
increase of 27.91% and 16.68% in mAP results for the entire dataset whilst being
17.6x and 3.6x smaller, respectively.
The convolutional features derived from the Efficient-Net were further enhanced
for the image retrieval task by leveraging R-MAC vectors which utilize an object
localization approach for improved query localization and recognition, and also a
diffusion process was carried on the resultant image manifold for refined image
ranking, this has shown to achieve considerable improvements in mAP calculations
when calculated for the entire dataset in particular there was a 17.29% increase.
