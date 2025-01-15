# SDeep 
## Requirements and Dependencies
- python 3.9
- torch 
- torchvision
- easydict
- nltk
- scikit-image
- At least 1x12GB NVIDIA GPU

## Installation
  ```
        pip install -r requirements.txt
        cd SDeep/code/
  ```

  
## Prepare Datasets
You can download images and the preprocessed metadata and put them to 'data'.
- images : the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and coco [coco2014](http://cocodataset.org/#download) 
- the preprocessed metadata : [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) and [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) 

## Download pre-train model
We provide the pre-train model, you can download our checkpoint to test. [pre-train](https://pan.baidu.com/s/1fJJvURofSG6-N5D4IuVK9g).

## Testing pipelines
You can synthesize images from the input text.
  ```
        cd src
        python test.py 
  ```
## Training
  ```
        cd src
        python train.py 
  ```

## Results 
Example results on the CUB and COCO.
<div align="center">
  <img src=https://github.com/zxcnmmmmm/SDeep/blob/main/github-domo.jpg>
</div>

