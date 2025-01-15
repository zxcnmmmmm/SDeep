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
- CUB dataset  
You can download [bird images](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and put them to 'data/birds/'.  
You can download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) and extract them to 'data/'  

- COCO dataset  
You can download [coco images](http://cocodataset.org/#download) and put them to 'data/coco/images/'.  
You can download the preprocessed metadata for [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to 'data/'  

## Download pretrained model
We provide the pre-train model, you can download our checkpoint to test.  
- The pretained model for [CUB](https://pan.baidu.com/s/1fJJvURofSG6-N5D4IuVK9g).  
- The pretained model for [coco](https://pan.baidu.com/s/1fJJvURofSG6-N5D4IuVK9g).

## Testing pipelines
In test.py, specify the path to the checkpoint using the path argument. Then, you can synthesize images from the input text.
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

