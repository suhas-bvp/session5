This project implements a basic Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch.

## Project Structure

├── data/
│ └── MNIST/
│ └── raw/
│ ├── t10k-images-idx3-ubyte
│ ├── t10k-images-idx3-ubyte.gz
│ ├── t10k-labels-idx1-ubyte
│ ├── t10k-labels-idx1-ubyte.gz
│ ├── train-images-idx3-ubyte
│ ├── train-images-idx3-ubyte.gz
│ ├── train-labels-idx1-ubyte
│ └── train-labels-idx1-ubyte.gz
└── mnist_cnn.py


- `mnist_cnn.py`: Contains the CNN model definition, data loading, training, and evaluation logic.
- `data/`: This directory will store the downloaded MNIST dataset.

## Features

- **Simple CNN Architecture**: A straightforward CNN model with two convolutional layers, max-pooling, and fully connected layers.
- **PyTorch Implementation**: Built entirely using the PyTorch deep learning framework.
- **MNIST Dataset**: Utilizes the popular MNIST dataset for handwritten digit classification.
- **Train and Evaluate**: Includes functions for training the model and evaluating its performance on a test set.
- **GPU Support**: Automatically uses a CUDA-enabled GPU if available, otherwise falls back to CPU.

## Getting Started

### Prerequisites

- Python 3.12
- `pip` (Python package installer)
-  pip install torch torchvision

### Running the Project

To train and evaluate the CNN model, simply run the `mnist_cnn.py` script:

```bash
python mnist_cnn.py
```

The script will:
1. Download the MNIST dataset to the `./data` directory (if not already present).
2. Initialize and print the number of trainable parameters in the model.
3. Train the model for a specified number of epochs (default is 1).
4. Evaluate the trained model on the test set and print the final accuracy.

## Model Architecture

The `SimpleCNN` model consists of:
- **Input Layer**: Expects a single-channel (grayscale) image of size 28x28.
- **Convolutional Block 1**:
    - `nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)`
    - `nn.ReLU()` activation
    - `nn.MaxPool2d(kernel_size=2, stride=2)` (output size: 14x14 with 8 channels)
- **Convolutional Block 2**:
    - `nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)`
    - `nn.ReLU()` activation
    - `nn.MaxPool2d(kernel_size=2, stride=2)` (output size: 7x7 with 16 channels)
- **Fully Connected Layers**:
    - `nn.Linear(16 * 7 * 7, 28)`
    - `nn.ReLU()` activation
    - `nn.Dropout(0.5)` for regularization
    - `nn.Linear(28, 10)` (output for 10 classes: digits 0-9)

## Configuration

The following hyperparameters can be adjusted in `mnist_cnn.py`:

- `BATCH_SIZE`: Number of samples per batch during training (default: 64).
- `LEARNING_RATE`: Learning rate for the Adam optimizer (default: 0.001).
- `EPOCHS`: Number of training epochs (default: 1).

## License

This project is open-source and available under the MIT License.