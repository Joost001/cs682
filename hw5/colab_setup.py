from google import drive
drive.mount('/content/gdrive')

import os
os.chdir("gdrive/My Drive/my-dl-assignment-google-drive-folder")

#!pip3 install torch torchvision

import torch
a = torch.Tensor([1]).cuda()
print(a)