# Ewe Audio Translation Baseline Model (CNN)

This repository contains code to train a Convolutional Neural Network (CNN) model for audio classification as part of the Zindi Ewe Audio Translation competition. This project is a group project for our **Advanced Machine Learning** class, we are a team of six students working together under the group name **Astronormers**.

## Features
- **Data Preprocessing**: Converts audio files from time-domain to frequency-domain using FFT.
- **Normalization**: Data is normalized based on the results of the Shapiro-Wilk test.
- **Model Architecture**: A custom CNN is designed with 1D convolutional layers followed by fully connected layers.
- **Training Pipeline**: Implements a PyTorch-based training loop using cross-entropy loss and the Adam optimizer.

## Dataset
- **Audio Files**: `.wav` files containing Ewe audio samples.
- **Labels**: Corresponding class labels for each audio sample.

## Preprocessing Steps
- Audio files are loaded and transformed using Fast Fourier Transform (FFT).
- The frequency spectrum is downsampled and normalized to a range of 0-1.

## Baseline Model Architecture
The model includes:
- Two convolutional layers with ReLU activations and max-pooling.
- Fully connected layers for classification.

## Training
The model is trained for 100 epochs using:
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam optimizer
- **Metrics**: Accuracy is computed for each batch.

## Group Information
This project is a collaborative effort by the group **Astronormers**, composed of six students, as part of the **Advanced Machine Learning** course.
Team members:
-Micha
-Brandon
-Ridwan
-Pragya
-Boluwarin
-Yusuf

## Usage
1. Clone the repository and navigate to the directory.
2. Install dependencies
3. Run the training Code
   

## Acknowledgements
This project is part of the Zindi Ewe Audio Translation competition and will be completed as part of the Advanced Machine Learning course. Special thanks to our group **Astronormers** for their collaborative effort.
