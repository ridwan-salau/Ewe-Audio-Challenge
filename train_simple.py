import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # for latent dims > 2
from torchvision import datasets, transforms
import random
import shutil
import datetime
import time
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

RANDOM_SEED = 42

def seed_everything(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AudioDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, duration=2):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.target_size = (128, 128)  # Fixed size for mel spectrograms
        self.top_db = 15  # Threshold for silence detection
        
        # Get all audio files
        self.files = []
        self.labels = []
        self.file_ids = []
        for class_idx, class_dir in enumerate(sorted(os.listdir(data_dir))):
            class_path = self.data_dir / class_dir
            if class_path.is_dir():
                for file in class_path.glob('*.wav'):
                    self.files.append(file)
                    self.labels.append(class_path.name)
                    self.file_ids.append(file.stem)
            elif class_path.is_file() and class_path.suffix == '.wav':
                self.files.append(class_path)
                self.labels.append(-1)
                self.file_ids.append(class_path.stem)

        self.unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to numpy for librosa processing
        waveform_np = waveform.numpy().squeeze()
        
        # Trim silence from both ends
        trimmed_waveform, trim_indexes = librosa.effects.trim(
            waveform_np,
            top_db=self.top_db,
            frame_length=2048,
            hop_length=512
        )
        
        # Convert back to torch tensor
        waveform = torch.from_numpy(trimmed_waveform).unsqueeze(0)
        
        # Ensure minimum length (pad if necessary)
        min_samples = int(self.sample_rate * self.duration)
        if waveform.size(1) < min_samples:
            waveform = torch.nn.functional.pad(
                waveform, 
                (0, min_samples - waveform.size(1))
            )
        
        # If longer than duration, take a random segment
        if waveform.size(1) > min_samples:
            max_start = waveform.size(1) - min_samples
            start = torch.randint(0, max_start, (1,))
            waveform = waveform[:, start:start + min_samples]
                
        # Convert to mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        mel_spec = mel_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # After creating mel_spec, resize it to fixed dimensions
        mel_spec = mel_spec.squeeze(0)  # Remove channel dimension temporarily
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
        # mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        label = self.label_to_idx[self.labels[idx]]
        return mel_spec, torch.tensor(label)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Encoder
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 1x128x128 -> 32x64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), # 32x64x64 -> 16x32x32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16x32x32 -> 32x16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x16x16 -> 64x8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64x8x8 -> 128x4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=4), # 128x4x4 -> 128x1x1
            nn.Flatten(),
            nn.Linear(128, 32), # 128x1x1 -> 32
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 8) # 32 -> 8
        )
    
    def forward(self, x):
        return self.model(x)

# Weight initialization
def init_weights():
    model = CNN()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
    return model

def train():
        # Hyperparameters
    batch_size = 128
    learning_rate = 1e-2
    num_epochs = 100
    latent_dim = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create dataset and split into train/val
    dataset = AudioDataset("train_data")
    train_size = int(0.8 * len(dataset))  # 80% train, 20% val
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model and optimizer
    model = init_weights().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    start_time = time.time()
    epoch_start_time = start_time
    batch_start_time = start_time
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 40 == 0:
                batch_duration = time.time() - batch_start_time
                epoch_duration = time.time() - epoch_start_time
                total_duration = time.time() - start_time
                
                log = (f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                       f'LR: {scheduler.get_last_lr()[0]:.6f}\n'
                       f'Batch Duration: {batch_duration:.2f}s, '
                       f'Epoch Duration: {epoch_duration:.2f}s, '
                       f'Total Duration: {total_duration:.2f}s')
                print(log)
                with open(run_dir / 'log.txt', 'a') as f:
                    f.write(log + '\n')
        
        # Step the scheduler
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}')
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        # Add these to collect predictions and true labels
        all_predictions = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Get predictions and scores
                scores = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Save if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'index_to_label': dataset.idx_to_label,
            }, run_dir / 'best_model.pt')
        
        # Update training plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(range(1, epoch + 2), train_losses, 'b-', label='Training Loss')
        ax1.plot(range(1, epoch + 2), val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(range(1, epoch + 2), val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(run_dir / 'training_metrics.png')
        plt.close()
        
        # Log metrics
        log = (f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
               f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, '
               f'LR: {scheduler.get_last_lr()[0]:.6f}')
        print(log)
        with open(run_dir / 'log.txt', 'a') as f:
            f.write(log + '\n')

        # Plot confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=dataset.unique_labels,
                   yticklabels=dataset.unique_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(run_dir / f'confusion_matrix_epoch_{epoch}.png')
        plt.close()

        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # One-vs-Rest ROC curves
        for i, class_name in enumerate(dataset.unique_labels):
            # Convert to binary classification problem
            binary_labels = (all_labels == i).astype(int)
            class_scores = all_scores[:, i]
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(binary_labels, class_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(run_dir / f'roc_curves_epoch_{epoch}.png')
        plt.close()

def test_submission(run_dir):
    device = torch.device("cpu")
    batch_size = 128
    test_dataset = AudioDataset("test_data")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CNN().to(device)
    checkpoint = torch.load(run_dir / 'best_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    index_to_label = checkpoint['index_to_label']

    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs: torch.Tensor = model(data)
            predictions.extend(outputs.argmax(dim=1).tolist())
    
    predictions = [index_to_label[pred] for pred in predictions]
    with open(run_dir / 'submission.csv', 'w') as f:
        f.write('id,class\n')
        for id, label in zip(test_dataset.file_ids, predictions):
            f.write(f'{id},{label}\n')
    
def validation_loop(run_dir):
    # Hyperparameters
    batch_size = 128
    device = torch.device("cpu")
    
    # Create dataset and split into train/val
    dataset = AudioDataset("train_data")
    train_size = int(0.8 * len(dataset))  # 80% train, 20% val
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model and criterion
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint
    checkpoint = torch.load(run_dir / 'best_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Validation phase
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    
    # Add these to collect predictions and true labels
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
            # Get predictions and scores
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset.unique_labels,
                yticklabels=dataset.unique_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(run_dir / 'confusion_matrix_epoch.png')
    plt.close()

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # One-vs-Rest ROC curves
    for i, class_name in enumerate(dataset.unique_labels):
        # Convert to binary classification problem
        binary_labels = (all_labels == i).astype(int)
        class_scores = all_scores[:, i]
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(binary_labels, class_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(run_dir / 'roc_curves_epoch.png')
    plt.close()

    return avg_val_loss, val_accuracy, all_predictions, all_labels, all_scores
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--run_dir', type=Path, default='runs_simple')
    args = parser.parse_args()

    seed_everything()
    if args.test:
        test_submission(args.run_dir)
        exit()

    if args.validation:
        validation_loop(args.run_dir)
        exit()

    run_dir = Path(args.run_dir)
    run_id = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = run_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # Save train_simple.py in run_dir
    shutil.copy('train_simple.py', run_dir / 'train_simple_checkpoint.py')


    train()
    test_submission(run_dir)