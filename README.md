# Detectron2 Cats Vs Dogs
Train a detectron model on a new dataset. (on coco format dataset)

## Introduction
* We use catvsdogs dataset which only has two classes: cats and dogs. We'll train a segmentation model from an existing model pre-trained on COCO dataset, available in detectron2's model zoo.

## How To Create Custom Dataset
* First, you need to get a labelling tool such as Labelme. You can insall [labelme](https://github.com/wkentaro/labelme) using the link.
* Every individual object in an image must be classified and identified each instance of an object within an image. More information : [Image Segmentation](https://datagen.tech/guides/image-annotation/labelme/)

## How To Convert Labelme Annotation Files to COCO Format
You can run ```labelme2coco.py``` script to generate a COCO data format file for both train and validation dataset.
```
python labelme2coco.py {path_to_train_folder} --output {path_to_train_folder/via_region_data.json}
```
```
python labelme2coco.py {path_to_validation_folder} --output {path_to_validation_folder/via_region_data.json}
```
