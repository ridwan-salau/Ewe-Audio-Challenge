import torch
import torchaudio
import torchaudio.transforms as T
from torch.functional import F
import scipy.signal as ss
import numpy as np

class FFTransform:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Apply the Fast Fourier Transform (FFT)
        fft_result = torch.fft.fft(waveform)

        # Compute the magnitude spectrum (optional)
        fft_centered = torch.fft.fftshift(fft_result)
        magnitude_spectrum = torch.log10(fft_centered**2)
        
        return magnitude_spectrum
        
class STFTTransform:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        stft_result = torch.stft(waveform, n_fft=2048, hop_length=512, win_length=2048, window=torch.hann_window(2048), center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        return stft_result

class TrimSilence:
    def __init__(self, sample_rate: int, trigger_level: float = 0.01):
        self.sample_rate = sample_rate
        self.trigger_level = trigger_level

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Convert torch tensor to numpy array
        audio = waveform.numpy().squeeze()
        
        # Calculate the absolute value of the signal
        abs_audio = np.abs(audio)
        
        # Find the indices where the signal is above the threshold
        non_silent_indices = np.where(abs_audio > self.trigger_level)[0]
        
        if len(non_silent_indices) == 0:
            return waveform
            
        # Get the start and end indices with padding
        padding = int(self.sample_rate * 0.1)  # 100ms padding
        start = max(0, non_silent_indices[0] - padding)
        end = min(len(audio), non_silent_indices[-1] + padding)
        
        # Trim the audio
        trimmed_audio = audio[start:end]
        
        # Convert back to torch tensor
        return torch.from_numpy(trimmed_audio).unsqueeze(0)

class OneHot:
    def __init__(self, CLASS_LABELS_MAP: dict):
        self.num_classes = len(CLASS_LABELS_MAP)
        self.CLASS_LABELS_MAP = CLASS_LABELS_MAP

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        return F.one_hot(torch.tensor(self.CLASS_LABELS_MAP[label]), num_classes=self.num_classes)

class Resample:
    def __init__(self, num: int):
        self.num = num

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[1] > waveform.shape[0]:
            waveform = waveform.T
        waveform = ss.resample(waveform, self.num, axis=0)
        return torch.from_numpy(waveform)

def collate_fn(batch):
    waveforms, labels = zip(*batch)

    # Right zero-pad all one-hot text sequences to max input length
    max_input_len = torch.max(torch.LongTensor([x.shape[1] for x in waveforms]))
    min_input_len = torch.min(torch.LongTensor([x.shape[1] for x in waveforms]))
    waveforms = [F.pad(x.mean(dim=0), (0, max_input_len - x.shape[1]), "constant", 0) for x in waveforms]

    return torch.stack(waveforms), torch.stack(labels)
    