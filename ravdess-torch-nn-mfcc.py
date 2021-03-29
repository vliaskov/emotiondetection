# Taken from https://www.kaggle.com/kuntaldas599/emotional-speech-classification2d

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
speech_dir = '/mnt/devel/datasets-psychology/ravdess'
song_dir = '../input/ravdess-emotional-song-audio'
#x = [dir for dir in os.listdir(speech_dir) if dir.startswith('Actor_')]



def load_RAVDESinfo(data_dir, list_classes, X=None, Y=None):
  """
  this function will return audio file PATH and labels seperately when data directory(`data_dir`) 
  and the class list(list_classes) are passed to it
  """
  audio_dataset = list()

  if not(X or Y):
    X = list()
    Y = list()
  actors = [dir for dir in os.listdir(data_dir) if dir.startswith('Actor_')]
  for dir in actors:
    act_dir = os.path.join(data_dir,dir)
    for wav in os.listdir(act_dir):
      # getting labels form the encoded file names
      label = [(int(i)-1) for i in wav.split('.')[0].split('-')]
      # converting gender labels to only 0 and 1
      label[-1] = 1 if label[-1]%2 else 0

      l_text = []

      # converting labels back to string
      for i in range(0, len(label)):
        l_text.append(list_classes[i][label[i]])
        
      # excluding nutral, disgust and surprise Emotion
      if(l_text[2] == "neutral" or l_text[2] == "disgust" or l_text[2] == "surprised"):
        continue
      X.append(os.path.join(act_dir, wav))
      Y.append(l_text)
  
  return X,Y


speech_info = load_RAVDESinfo(speech_dir,classes_list)
print(f"length of : files - {len(speech_info[0])}, labels - {len(speech_info[1])}")

print(speech_info[0][14], speech_info[1][14])




def test_trainSplit(dir_list,class_list,test_size=0.3):
  """
  this function given the list of directories `dir_list` and `test_size` writes out 
  `test.csv` and `train.csv` with the help of the previous function `load_RAVDESinfo`. 
  the function will not overwrite anyfile if present in the same directory.
  """
  if os.path.isfile('test.csv') or os.path.isfile('train.csv'):
    print ("csv files exist")
    return
  
  print ("csv files do not exist\n creating Test, Train Dataset(train.csv, test.csv)")
  X = None
  Y = None
  for dir in dir_list:
    X,Y = load_RAVDESinfo(dir,class_list,X,Y)

  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=42)

  test_audios = [] 
  train_audios = [] 
  
  label_headers = ['Audio_file', 'modality', 'vocal_channel', 'emotion', 'emotional_intensity', 'statement', 'repetition', 'gender']
  
  with open('train.csv', 'w+') as train:
    train_write = csv.writer(train)
    train_write.writerow(label_headers)
    for audio, label in zip(X_train, Y_train):
      row = [audio]
      row.extend(label)
      train_write.writerow(row)

  with open('test.csv', 'w+') as test:
    test_write = csv.writer(test)
    test_write.writerow(label_headers)
    for audio, label in zip(X_test, Y_test):
      row = [audio]
      row.extend(label)
      test_write.writerow(row)        

test_trainSplit([speech_dir],classes_list,test_size=0.3)
#320


test_csv = 'test.csv'
train_csv = 'train.csv'

test_dataframe = pd.read_csv(test_csv)
train_dataframe = pd.read_csv(train_csv)

print(len(test_dataframe), len(train_dataframe))
print(test_dataframe.sample(3))

fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.title('Count of Emotions in test DATA', size=16)
sns.countplot(x=test_dataframe.emotion)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
# plt.show()

plt.subplot(1,2,2)
plt.title('Count of Emotions in train DATA', size=16)
sns.countplot(x=train_dataframe.emotion)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()

def load_mono(audio_path):
  wave, sr = librosa.load(audio_path,sr=None,mono=True)
  
  # trim silent edges
  wave, _ = librosa.effects.trim(wave)

  # convert to tensor
  wave = torch.FloatTensor(wave).unsqueeze(0)
  return wave, sr

def show_audio(audio_path):
  y, sr = librosa.load(audio_path, sr=None,mono=True)
  print(f"Sample rate : {sr}")
  
  # trim silent edges
  audio, _ = librosa.effects.trim(y)
  
  fig = plt.figure(figsize=(20,15))
  n_fft = 2048
  hop_length = 256
  n_mels = 128

  plt.subplot(3,3,1)
  librosa.display.waveplot(audio, sr=sr);
  plt.title('1. raw wave form data');

  plt.subplot(3,3,2)
  D = np.abs(librosa.stft(audio[:n_fft], n_fft=n_fft, hop_length=n_fft+1))
  plt.plot(D);
  plt.title(f'2. fourier transform of a window(length={n_fft})');

  plt.subplot(3,3,3)
  D = np.abs(librosa.stft(audio, n_fft=n_fft,  hop_length=hop_length))
  librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear');
  plt.colorbar();
  plt.title('3. applyed the Fourier Transform');

  plt.subplot(3,3,4)
  DB = librosa.amplitude_to_db(D, ref=np.max)
  librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
  plt.colorbar(format='%+2.0f dB');
  plt.title('4. Spectrogram');

  mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
  
  plt.subplot(3,3,5);
  librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='linear');
  plt.ylabel('Mel filter');
  plt.colorbar();
  plt.title('5. Our filter bank for converting from Hz to mels.');

  plt.subplot(3, 3, 6);
  mel_10 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=10)
  librosa.display.specshow(mel_10, sr=sr, hop_length=hop_length, x_axis='linear');
  plt.ylabel('Mel filter');
  plt.colorbar();
  plt.title('6. Easier to see what is happening with only 10 mels.');

  plt.subplot(3, 3, 7);
  idxs_to_plot = range(0,127,10)
  for i in idxs_to_plot:
      plt.plot(mel[i]);
  plt.legend(labels=[f'{i+1}' for i in idxs_to_plot]);
  plt.title('6. Plotting some of the triangular filters from the mels');

  plt.subplot(3,3,8)
  plt.plot(D[:, 1]);
  plt.plot(mel.dot(D[:, 1]));
  plt.legend(labels=['Hz', 'mel']);
  plt.title('8. One sampled window for example, before and after converting to mel.');

  plt.subplot(3,3,9)
  S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
  S_DB = librosa.power_to_db(S, ref=np.max)
  librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
  plt.colorbar(format='%+2.0f dB');
  plt.title('9. Mel - Spectrogram');

  fig.tight_layout() 
  plt.show()


path = test_dataframe.iloc[20]["Audio_file"]
print(test_dataframe.iloc[20]["emotion"])
show_audio(path)

# Lets play the audio
#data, rate = load_mono(path)
#ipd.Audio(data=data.numpy(),rate=rate)




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



i = 20
audio_path = test_dataframe["Audio_file"].iloc[i]

SPECTROGRAM, audio, rate = load_spec(audio_path,mode=0)
print(test_dataframe["emotion"].iloc[i])
print(audio.shape)
print(SPECTROGRAM.shape)

plt.imshow(SPECTROGRAM.log10()[0,:,:].numpy(), cmap="inferno")
#ipd.Audio(data=audio,rate=rate)



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


# mode : 0-spectrogram, 1-mel_spectrogram, 2-mfcc

max_dim = get_maxDim(test_csv,mode=1)
print(max_dim)
max_dim = get_maxDim(train_csv, mode=1)
print(max_dim)

batch_size = 40

#train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
#test_dl = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


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



@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
#         for batch in tqdm(train_loader):
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader),
                                                pct_start=0.20)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
#         for batch in tqdm(train_loader):
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


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

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
#train_dl = DeviceDataLoader(train_dl, device)
#test_dl = DeviceDataLoader(test_dl, device)
#model = to_device(model,device)

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');

epochs = 50
max_lr = 1e-5
grad_clip = 0.1
weight_decay = 1e-5
opt_func = torch.optim.Adam


M = 2 # mfcc

mfcc_max_dim = get_maxDim(test_csv,mode=M)
mfcc_max_dim = get_maxDim(train_csv, *mfcc_max_dim, mode=M)

test_dataset = ravdessEmoDataSet(test_csv, dim = mfcc_max_dim, mode=M)
train_dataset = ravdessEmoDataSet(train_csv, dim = mfcc_max_dim, mode=M)
# train_dataset = ravdessEmoDataSet(train_csv, dim = max_dim, mode=0, loader=train_aug)
print(f" length : \n\ttest:{len(test_dataset)},\n\ttrain:{len(train_dataset)}")

batch_size = 40

train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

class_len= len(test_dataset.classes)
print(f"class lenght : {class_len}")

mfcc_model = EmotionalResnet18(1,class_len,pretrained=False)
# model.load_state_dict(torch.load("./EAC-mfcc_model.pth"))

device = get_default_device()
print(f"device : {device}")

train_dl = DeviceDataLoader(train_dl, device)
test_dl = DeviceDataLoader(test_dl, device)
mfcc_model = to_device(mfcc_model,device)


history = [evaluate(mfcc_model, test_dl)]
print(history)

history += fit_one_cycle(epochs, max_lr, mfcc_model, train_dl, test_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)


torch.save(mfcc_model.state_dict(), 'EAC-mfcc_model.pth')

fig = plt.figure(figsize=(15,7))

plt.subplot(2,2,1)
plot_accuracies(history)

plt.subplot(2,2,2)
plot_losses(history)

plt.subplot(2,2,3)
plot_lrs(history)

fig.tight_layout() 
plt.show()



