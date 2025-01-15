# SDeep 
## Requirements and Dependencies
- python 3.9
- torch 
- torchvision
- easydict
- nltk
- scikit-image
- At least 1x12GB NVIDIA GPU

## Install
 - Dependencies.
  ```
       pip install -r requirements.txt
  ```
- Download and install [CLIP](https://github.com/openai/CLIP).
  ```
       cd CLIP-main
       python3 setup.py build install
  ```

  
## Prepare Datasets
- CUB dataset  
You can download [bird images](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and put them to ```data/birds/```  
You can download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) and extract them to ```data/```  

- COCO dataset  
You can download [coco images](http://cocodataset.org/#download) and put them to ```data/coco/images/```  
You can download the preprocessed metadata for [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to ```data/```    

## Download pretrained model
We provide the pretained model, you can download our checkpoints to test.  
- The pretained model for [bird](https://pan.baidu.com/s/1fJJvURofSG6-N5D4IuVK9g). Download and save it to  ```./code/saved_models/bird/```  
- The pretained model for [coco](https://pan.baidu.com/s/1fJJvURofSG6-N5D4IuVK9g). Download and save it to  ```./code/saved_models/coco/```  

## Testing pipelines
Specify the ```path``` to the checkpoint in test.py. Then, you can synthesize images from the input text. The results saved in ```./samples``` 
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

