import numpy as np
import torch
import scipy.io.wavfile as sw
import scipy.signal as ss
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import scipy.stats as stat
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torchaudio
import torch
import sys
import random

from pathlib import Path
from models.base import init_weights as base_init_weights
# from transforms import TrimSilence, STFTTransform, OneHot, collate_fn, Resample

from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument('--val_only', action='store_true', default=False)
args = args.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
# Load or preprocess data
if not (os.path.exists('./training_data.npy') and os.path.exists('./training_labels.npy')):
    # Data preprocessing code here
    
    # import .wav data & labels
    img_root = "./TechCabal Ewe Audio Files/"
    csv_data = pd.read_csv('./Train.csv', sep=',')
    print(csv_data.keys())
    images = csv_data['audio_filepath']
    labels = csv_data['class']

    unique_labels = np.unique(labels)
    print(unique_labels)
    labels_class = np.arange(0,len(np.unique(labels)))
    print(labels_class)

    # create class to load data into DataLoader (and preprocess)
    freq_spectrum = []
    freq_labels = []
    trim_silence = TrimSilence(sample_rate=48_000)
    start_time = time.time()
    print(f"Processing {len(images)} audio files")
    for widx, wv in enumerate(images):
        fpath = os.path.join(img_root, wv)
        if os.path.isfile(fpath) == 1:
            fs, data = sw.read(fpath)
            class_tmp = labels[widx]
            cidx = np.where(class_tmp == unique_labels)[0]
            freq_labels.append(cidx)
            # convert time domain to frequency domain
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            # transforming the 1-D time-series into a frequency spectrum
            fft = np.fft.fft(data)
            fft_centered = np.fft.fftshift(fft)
            fft_magn = np.log10(np.abs(fft_centered)**2)
            # print(fft_magn.shape)
            fft_magn_dwn = ss.resample(fft_magn, 51744, axis=0).astype(np.float32)

            freq_spectrum.append(fft_magn_dwn.T)

    print("Applying trim silence")
    training_data = np.stack(freq_spectrum, axis=0)
    print(training_data.shape)
    # training_data = trim_silence(torch.tensor(training_data)).numpy()
    min_len = training_data.min(axis=1)

    training_labels = np.array(freq_labels)

    print(f"Done processing data in {time.time() - start_time:.2f} seconds")

    imgs_length = [freq.shape[0] for freq in training_data]
    print(np.min(imgs_length))  


    np.save('training_data.npy', freq_spectrum)
    np.save('training_labels.npy', freq_labels)

# Load the training data
training_data = np.load('./training_data.npy', allow_pickle=True)
training_labels = np.load('./training_labels.npy', allow_pickle=True)

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(
    training_data, training_labels, test_size=0.2, random_state=42
)

# Normalize training and validation data
train_data = (train_data - np.min(train_data)) / (np.max(train_data) - np.min(train_data))
val_data = (val_data - np.min(val_data)) / (np.max(val_data) - np.min(val_data))

# One-hot encode training and validation labels
train_labels_tmp = torch.tensor(train_labels.flatten())
train_labels = nn.functional.one_hot(train_labels_tmp)
val_labels_tmp = torch.tensor(val_labels.flatten())
val_labels = nn.functional.one_hot(val_labels_tmp)


# Dataset and DataLoader
class MyDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

print("Data Shape: ", train_data.shape, train_labels.shape)
# Create training and validation datasets and dataloaders
train_dataset = MyDataset(train_data, train_labels)
val_dataset = MyDataset(val_data, val_labels)
dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=False)
dataloader_val = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=False)

# Create and initialize the model
model = base_init_weights(n_classes=8)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

def train():
    # Training loop
    epochs = 150
    best_val_accuracy = 0
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        for i, (inputs, labels) in enumerate(dataloader_train):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.squeeze(outputs).float()
            labels = labels.float()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            softmax = nn.Softmax(dim=1)(outputs)
            train_accuracy += (torch.argmax(softmax, dim=1) == torch.argmax(labels, dim=1)).float().mean().item()
        
        train_loss /= len(dataloader_train)
        train_accuracy /= len(dataloader_train)
        
        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloader_val:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = torch.squeeze(outputs).float()
                labels = labels.float()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                softmax = nn.Softmax(dim=1)(outputs)
                val_accuracy += (torch.argmax(softmax, dim=1) == torch.argmax(labels, dim=1)).float().mean().item()
        
        val_loss /= len(dataloader_val)
        val_accuracy /= len(dataloader_val)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Early stopping
        # if epoch > 20 and val_loss > min(val_losses[-20:]):
        #     print("Early stopping")
        #     break

        val_losses.append(val_loss)

def evaluate():
    # Load the best model checkpoint
    print("\nEvaluating best model checkpoint...")
    best_model = base_init_weights(n_classes=8)
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.to(device)
    best_model.eval()

    # Validation loop with best model
    val_accuracy = 0
    val_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            outputs = torch.squeeze(outputs).float()
            labels = labels.float()
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            softmax = nn.Softmax(dim=1)(outputs)
            batch_accuracy = (torch.argmax(softmax, dim=1) == torch.argmax(labels, dim=1)).float().mean().item()
            val_accuracy += batch_accuracy
            
            predictions.extend(torch.argmax(softmax, dim=1).cpu().numpy())
            true_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())

    val_loss /= len(dataloader_val)
    val_accuracy /= len(dataloader_val)

    print(f"\nBest Model Performance:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

if __name__ == '__main__':
    if args.val_only:
        evaluate()
    else:
        train()
        evaluate()