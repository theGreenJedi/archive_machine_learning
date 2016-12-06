# ImageNet classification with AlexNet

AlexNet is a convolutional neural network that consists of 5 convolutional layers and 3 fully-connected layers.
This neural network was first proposed by Alex Krizhevsky, *et al.* [[1]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), which won the first place at ILSVRC 2012, achieving a top-5 error rate of 15.3%. 

This code represents that how to classify an image using pre-trained AlexNet with ILSVRC 2012 dataset. *ILSVRC* (Imagenet Large Scale Visual Recognition Challenge) is an annual competition for image classification using many (~million) images in 1,000 categories. 

[](./image/cat.png)

In ```alexnet_classify.py```...<br>
Since training the whole process with AlexNet requires very large resources and much time, we just load a weight set pre-trained by AlexNet. And We read an image, and perform image classification that outputs 5 most probable categories and their probabilities. 

## Brief description of code
1. Load an image 'cat'
   * we load a test image 'cat.jpg' and resize it to 256x256. If the image is not RGB format, then we convert it to an RGB image.
   * the shape of loaded data is (256, 256, 3)
2. Cropping the loaded image
   * we obtain 10 images of size 227x227 by shifting and cropping the image
   * the shape of cropped and loaded image is (10, 227, 227, 3)
3. Subtract mean to the result of 2
   * subtracting the dataset mean serves to "center" the data
   * this is a kind of normalization. It helps for a model to have good local optimums. 
   * since the weights of AlexNet are trained given data which subtracted by mean, we also subtract mean to test data (=cat)
4. Load a weight set pre-trained by AlexNet
   * we load ```bvlc_alexnet.npy``` as a numpy array 
5. Construct the AlexNet
   * among five number of convolutional layers, we don't have maxpooling in two number of convolutional layers.
6. Evaluate the AlexNet
   * we don't need to train the model
   
   
   
<br>
> EE488C Special Topics in EE Deep Learning and AlphaGo, Fall 2016

