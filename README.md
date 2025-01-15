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
 - ### Clone this repo and install dependencies.
  ```
       git clone https://github.com/zxcnmmmmm/SDeep
       pip install -r requirements.txt
  ```
- ### Download and install [CLIP](https://github.com/openai/CLIP).
```
      cd CLIP-main
      python3 setup.py build install
  ```
  
## Prepare Datasets
- ### CUB dataset  
You can download [bird images](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and put them to ```data/birds/```  
You can download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) and extract them to ```data/```  

- ### COCO dataset  
You can download [coco images](http://cocodataset.org/#download) and put them to ```data/coco/images/```  
You can download the preprocessed metadata for [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to ```data/```    

## Download pretrained models
### We provide the pretained models, you can download our checkpoints to test.  
- The pretained model for [bird](https://pan.baidu.com/s/1fJJvURofSG6-N5D4IuVK9g). Download and save it to  ```./code/saved_models/bird/```  
- The pretained model for [coco](https://pan.baidu.com/s/1fJJvURofSG6-N5D4IuVK9g). Download and save it to  ```./code/saved_models/coco/```  

## Testing pipelines
### Specify the ```path``` to the checkpoint in test.py. Then, you can synthesize images from the input text. 
  ```
       cd SDeep/code/src/
       python test.py 
  ```
Results are saved to ```./samples``` 

## Training
### Train SDeep models. ```*.yml``` files are example configuration files for training/evaluation our models.
- Train for CUB.
  ```
       cd SDeep/code
       bash src/train.sh ./cfg/bird.yml
  ```
- Train for COCO.
  ```
       cd SDeep/code
       bash src/train.sh ./cfg/coco.yml
  ```
- Resume training process  
If your training process is interrupted unexpectedly, set ```resume_epoch``` and ```resume_model_path``` in train.sh to resume training.

## Results 
### Example results on the CUB and COCO.
<div align="center">
  <img src=https://github.com/zxcnmmmmm/SDeep/blob/main/github-domo.jpg>
</div>

## Citing SDeep
If you find SDeep useful in your research, please consider citing:
  ```
  @article{ccc2025SDeep,
     author = {Wenli Chen, Yaqi Sun, Paul L. Rosin, YuKun Lai},
     title = {Stacked Deep Fusion GAN for Enhanced Text-to-Image Generation},
     Year = {2025},
     booktitle = {{The Visual Computer}}
  }
  ```
## Reference
- [GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis](https://arxiv.org/abs/2301.12959) [[code]](https://github.com/tobran/GALIP)
- [GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis](https://arxiv.org/abs/2301.12959) [[code]](https://github.com/tobran/GALIP)
