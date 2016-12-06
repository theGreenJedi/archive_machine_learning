# Fine-tuning AlexNet using Caltech101dataset

We will re-train the last fully-connected layer of the pre-trained AlexNetto classify new categories of images from Caltech 101 dataset. Caltech 101 dataset contains a set of images in 101 categories such as airplanes, faces, and motorbikes, collected in 2003 by Fei-FeiLi, et al. There are about 40 to 800 images per category. Since the numbers of images vary a lot, we choose 10 categories with about 100 images in each category.The chosen 10 categories and associated image file names can be found in “101_labels_ten.txt”.

<br>

There are materials we need to fine-tuning.
---------------------
File Name | Description
----------|---------------
101_labels_ten.txt | image file names for 10 categories
crop_batch.py | cropping and data augmentation
divide_train_val.py | divide images into train and validation sets
Generate_HDF5.py | generate HDF5 files
ilsvrc_2012_mean.npy | mean information for ILSVRC 2012 dataset
project2_alexnet_finetune.py | code for fine-tuning
project2_HDF5_generation.py | code for generating *.h5 files

And we need Caltech 101 dataset which can be downloaded as follows:
```
$ wget--no-check-certificate https://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
$ tar xvzf101_ObjectCategories.tar.gz
```


