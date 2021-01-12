# MASKRCNN-CLEVR



## Set up environment



```sh
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch
pip install opencv-python
pip install pycocotools
```

## Training a new model

Hyperparameters:

category=True #25 categories or just 1 category)

batch_size = 2

test_split = 5 # how many images for testing set

root = "output" # dataset dir

num_epochs = 10

```sh
python train_clevr.py
```

visualize

```sh
python test_clevr.py
```

# MaskRCNN_for_CLEVR_dataset
