# Detectron2 Cats Vs Dogs :cat: :dog:
Train a detectron model on a custom dataset (Coco formated). 

## Introduction
* We use catvsdogs dataset which only has two classes: cats and dogs. We'll train a segmentation model from an existing model pre-trained on COCO dataset, available in detectron2's model zoo.
* Following the steps below you can create your dataset and train your instance segmentation model on detectron2.

## How To Create Custom Dataset
* First, you need to get a labelling tool such as Labelme. You can insall [labelme](https://github.com/wkentaro/labelme) using the link.
* Every individual object in an image must be classified and identified each instance of an object within an image. More information : [Image Segmentation](https://datagen.tech/guides/image-annotation/labelme/)
* Dataset folders must be like in the following format. [Example Dataset](https://github.com/mharunturkmenoglu/detectron2_catsvsdogs/blob/main/catsvsdogs.zip)
```bash
.
├── train/
│   ├── images
└── evaluation/
        ├── images   
```

## How To Convert Labelme Annotation Files to COCO Format
Run the ```labelme2coco.py``` script to generate a COCO data format file for both train and validation dataset as below. 
```
python labelme2coco.py {path_to_train_folder} --output {path_to_train_folder/via_region_data.json}
```
```
python labelme2coco.py {path_to_validation_folder} --output {path_to_validation_folder/via_region_data.json}
```
* Folder structure after conversion.
```bash
.
├── train/
│   ├── images
│   ├── via_region_data.json
└── evaluation/
        ├── images   
        ├── via_region_data.json
```
## How To Run Model Train
* Zip your dataset containing folder formatted like above.
* Upload the jupyter notebook to [Google Colab](https://colab.research.google.com/).
* Upload your dataset to Google Colab.
* Update the code snippet below according to the dataset you've been created.

```
from detectron2.data.datasets import register_coco_instances
register_coco_instances("YourTrainDatasetName", {},"path to train.json", "path to train image folder")
register_coco_instances("YourTestDatasetName", {}, "path to test.json", "path to test image folder")
```
* An example code:
```
from detectron2.data.datasets import register_coco_instances
register_coco_instances("train_dataset", {}, "/content/catsvsdogs/train/via_region_data.json", "/content/catsvsdogs/train")
register_coco_instances("val_dataset", {}, "/content/catsvsdogs/validation/via_region_data.json", "/content/catsvsdogs/validation")
```
* Before you starting the training, please make sure to update value of ```cfg.MODEL.ROI_HEADS.NUM_CLASSES = [Update Here] ``` is the same as number of classes in the dataset. 
```
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.TEST = ("val_dataset",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two class (cats and dogs). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
```
* Run the notebook.
