import torch
import os
from PIL import Image
import clip
import os.path as osp
import os, sys
import torchvision.utils as vutils
sys.path.insert(0, '../')
from models.utils import load_model_weights,mkdir_p
from models.model import NetG, TEXT_ENCODER


device = 'cpu'
CLIP_text = "ViT-B/32"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.eval()
text_encoder = TEXT_ENCODER(clip_model).to(device)
netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)
path = '..'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)
batch_size = 8
noise = torch.randn((batch_size, 100)).to(device)
input_text = ['...']
mkdir_p('./test_result')
with torch.no_grad():
    for i in range(len(input_text)):
        text = input_text[i]
        tokenized_text = clip.tokenize([text]).to(device)
        sent_emb, word_emb = text_encoder(tokenized_text)
        sent_emb = sent_emb.repeat(batch_size,1)
        fake_imgs = netG(noise,sent_emb,eval=True).float()
        name = f'{input_text[i].replace(" ", "-")}'
        vutils.save_image(fake_imgs.data, 'test_result/%s.png'%(name), nrow=8, value_range=(-1, 1), normalize=True)