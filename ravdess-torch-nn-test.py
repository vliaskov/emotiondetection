import csv
import sys
import os
import random
import zipfile

from copy import deepcopy

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
# from torchvision.utils import make_grid
# from torch.utils.data import random_split

#from tqdm.notebook import tqdm # Visualize the progress per epoch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd  # To play sound in the notebook
from sklearn.model_selection import train_test_split
from random import randrange

classes_list = [
  ['full-AV', 'video-only','audio-only'],
  ['speech', 'song'],
  ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
  ['normal', 'strong'], # NOTE: There is no strong intensity for the 'neutral' emotion.
  ["Kids are talking by the door", "Dogs are sitting by the door"],
  ['1st repetition', '2nd repetition'],
  ["male", "female"],
]

import os
# data_dir = './ravdess-emotional-speech-audio'
#speech_dir = '/mnt/devel/datasets-psychology/ravdess'
speech_dir = '/data/datasets-psychology/ravdess'
#x = [dir for dir in os.listdir(speech_dir) if dir.startswith('Actor_')]
test_csv = 'test.csv'
train_csv = 'train.csv'

test_dataframe = pd.read_csv(test_csv)
train_dataframe = pd.read_csv(train_csv)


class ravdessEmoDataSet(Dataset):
  def __init__(self, csv_file, dim, loader=None, mode=0):
    self.dataframe = pd.read_csv(csv_file)
    self.classes = list(set(self.dataframe["emotion"]))
    # self.classes = list(set(self.dataframe["gender"]))
    self.loader = loader
    self.dim = dim
    self.mode = mode

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, index):
    row = self.dataframe.loc[index]
    audio_path,_,_, emotion, emo_intensity,_,_, gender = row
    
    label = self.classes.index(emotion)
    # label = self.classes.index(gender)

    s, waveform, rate = load_spec(audio_path,mode=self.mode)

    if self.loader:
      s = self.loader(s)
    else:
      _, height, width = list(s.shape)
      diff = [self.dim[0] - height, self.dim[1] - width]
      # pad on both side of the tensor by zeros
      pd1 = (diff[1]//2, diff[1]-diff[1]//2, diff[0]//2,diff[0]-diff[0]//2)
      s = F.pad(s, pd1, mode='constant', value=0)

    return s, label

def predict_emotion(audio_path, model, max_height=0, max_width=0, mode=0):
    s, waveform, rate = load_spec(audio_path, mode = mode)

    _, height, width = list(s.shape)
    
    diff = [(max_height - height), (max_width - width)]
    # pad on both side of the tensor by zeros
    pd1 = (diff[1]//2, diff[1]-diff[1]//2, diff[0]//2,diff[0]-diff[0]//2)
    s = F.pad(s, pd1, mode='constant', value=0)
    
    # Convert to a batch of 1
    device = get_default_device()
    xb = to_device(s.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    print(f"predicted : {train_dataset.classes[preds[0].item()]}")
    print(f"predicted : {test_dataset.classes[preds[0].item()]}")
    return waveform,rate
    # return preds[0].item()

class SpectrogramClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {},".format(result['lrs'][-1]) if 'lrs' in result else '', 
            result['train_loss'], result['val_loss'], result['val_acc']))

def load_spec(audio_path, mode=0):
  """
  takes audio path and mode to return various audio 2D representation with the 
  actual audio and sample rate as tensor

  use mode=1 to get melspectrogram
  and mode=2 to get mfcc
  Default mode=0 for Spectrogram
  """
  wave, sr = librosa.load(audio_path,sr=None,mono=True)
  # trim silent edges(below 60 db by default), change the threashold by passing `top_db`
  # The threshold (in decibels) below reference to consider as silence (default : 60 db)
  s, _ = librosa.effects.trim(wave,top_db=60)
  
  # convert to tensor
  wave = torch.FloatTensor(s).unsqueeze(0)
    
  # generate (mel)spectrogram / mfcc
  if(mode == 1):
    # s = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(wave)
    s = librosa.feature.melspectrogram(y=s, sr=sr, hop_length=512)
  elif(mode == 2):
    # s = torchaudio.transforms.MFCC(sample_rate=sr)(wave)
    s = librosa.feature.mfcc(y=s, sr=sr, n_mfcc=40)
  else:
    # s = torchaudio.transforms.Spectrogram()(wave)
    freqs, times, s = librosa.reassigned_spectrogram(y=s, sr=sr, hop_length=512)
    
  s = torch.FloatTensor(s).unsqueeze(0)
  return s, wave, sr

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_maxDim(csv_file,max_height=0,max_width=0,mode =0, verbose=False):
  min_width = min_height = 5e5
#   stdv = []
#   med = []
#   mean = [] 
    
  dataframe = pd.read_csv(csv_file)
  for i,path in enumerate(dataframe["Audio_file"]):

    spec,_,_ = load_spec(path,mode=mode)
    
#     # calc std
#     stdv.append(torch.std(spec))
#     med.append(torch.median(spec))
#     mean.append(torch.mean(spec))
    
    _, height, width = list(spec.shape) 
    if(height > max_height):
      max_height = height
      if verbose:
        print(f"{i} max height - {height}")
    if(width > max_width):
      max_width = width
      if verbose:
        print(f"{i} max width - {width}")
    
    # just printing 
    if(min_width > width):
      min_width = width
      if verbose:
        print(f"{i} min width - {min_width}")
    if(min_height > height):
      min_height = height
      if verbose:
        print(f"{i} min height - {min_height}")

#   stdv = torch.FloatTensor(stdv)
#   med = torch.FloatTensor(med)
#   mean = torch.FloatTensor(mean)
#   print(f"\tstdv : {torch.mean(stdv)},\n\tmedian : {torch.mean(med)}, \n\tmean: {torch.mean(mean)}")
  print(f"min-width : {min_width},\tmin-height : {min_height}")
  print(f"max-width : {max_width},\tmax-height : {max_height}")

  return max_height,max_width


class EmotionalResnet18(SpectrogramClassificationBase):
    def __init__(self, in_channels,num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = torchvision.models.resnet18(pretrained)
        
        # Replace the first layer
        self.network.conv1 = nn.Conv2d( 
            in_channels,
            self.network.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3
        )
        
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, 512)
        # new layer introduced
        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
        )
            

    def forward(self, xb):

        out = self.network(xb)
        return self.fc2(out)

class_len=5
model = EmotionalResnet18(1,class_len,pretrained=False)
print ("Test model", sys.argv[1])
model.load_state_dict(torch.load(sys.argv[1]))
model.eval()

# indx = 15
indx = randrange(len(test_dataframe))

audio_path = os.path.join(test_dataframe.iloc[indx]["Audio_file"])

print(f"Audio Location : {audio_path}")
print(f"Label : {test_dataframe.iloc[indx]['emotion']}")

spectro_max_dim = get_maxDim(test_csv,mode=2)
spectro_max_dim = get_maxDim(train_csv, *spectro_max_dim, mode=2)

test_dataset = ravdessEmoDataSet(test_csv, dim = spectro_max_dim, mode=2)
train_dataset = ravdessEmoDataSet(train_csv, dim = spectro_max_dim, mode=2)

device = get_default_device()
model = to_device(model, device)
wave = predict_emotion(audio_path,model, *spectro_max_dim)

#ipd.Audio(data=wave,rate=48000)
