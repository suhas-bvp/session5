This project implements a basic Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch, with 10k parameters and an accuracy of 99%.

- `mnist_cnn.py`: Contains the CNN model definition, data loading, training, and evaluation logic.
- `data/`: This directory will store the downloaded MNIST dataset.

## Model Architecture

<img width="861" height="532" alt="image" src="https://github.com/user-attachments/assets/5add744f-cd1b-4a64-841d-7d690a2d0474" />


The model consists of:
- **Input Layer**: Expects a single-channel (grayscale) image of size 28x28.
- **Convolutional Block 1**:
    - `nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)`
    - `nn.ReLU()` activation
    - `nn.MaxPool2d(kernel_size=2, stride=2)` (output size: 14x14 with 8 channels)
- **Convolutional Block 2**:
    - `nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)`
    - `nn.ReLU()` activation
    - `nn.MaxPool2d(kernel_size=2, stride=2)` (output size: 7x7 with 16 channels)
- **Convolutional Block 3**:
    - `nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)`
    - `nn.ReLU()` activation
    - `nn.MaxPool2d(kernel_size=2, stride=2)` (output size: 7x7 with 16 channels)
    - nn.MaxPool2d(kernel_size=2, stride=2), # Reduces image size by half (7x7 -> 3x3)
    - nn.Dropout2d(0.2),
- **Fully Connected Layers**:
    -  nn.Linear(32 * 3 * 3, 10) # Reduced output features from 32 to 10
      
## Model Parameters
| Layer Name | Formula | Calculation | Parameters |
| :--- | :--- | :--- | :--- |
| **Conv2d_1** | $(C_{in} \times K^2 + 1) \times C_{out}$ | $(1 \times 3^2 + 1) \times 8$ | 80 |
| **BatchNorm2d_1** | $2 \times C_{out}$ | $2 \times 8$ | 16 |
| **Conv2d_2** | $(C_{in} \times K^2 + 1) \times C_{out}$ | $(8 \times 3^2 + 1) \times 16$ | 1,168 |
| **BatchNorm2d_2** | $2 \times C_{out}$ | $2 \times 16$ | 32 |
| **Conv2d_3** | $(C_{in} \times K^2 + 1) \times C_{out}$ | $(16 \times 3^2 + 1) \times 32$ | 4,640 |
| **BatchNorm2d_3** | $2 \times C_{out}$ | $2 \times 32$ | 64 |
| **Linear** | $(I + 1) \times O$ | $(288 + 1) \times 10$ | 2,890 |
| **Total** | | | **8,890** |


**parameters calculations**

Convolutional layers: (input channels×kernel height×kernel width + 1) × output channels 

Fully connected layers: (input features + 1) ) × output features

BatchNorm2d: These layers have two learnable parameters per output channel: one for a scaling factor (γ) and one for a shifting factor (β). 
The total parameters are 2 times the number of output channels (C out).

-> The +1 accounts for the bias term.

## Model Output

<img width="1333" height="536" alt="image" src="https://github.com/user-attachments/assets/bcdcb9dd-f7ab-4337-90f4-4c5fbfc58fac" />



## Configuration
The following hyperparameters can be adjusted in `mnist_cnn.py`:
- `BATCH_SIZE`: Number of samples per batch during training (default: 64).
- `LEARNING_RATE`: Learning rate for the Adam optimizer (default: 0.0005) 
- `EPOCHS`: Number of training epochs (default: 15).

### Prerequisites
- Python 3.12
- `pip` (Python package installer)
-  pip install torch torchvision

### Running the Project
python mnist_cnn.py

The script do:
1. Download the MNIST dataset to the `./data` directory (if not already present).
2. Initialize and print the number of trainable parameters in the model.
3. Train the model for a specified number of epochs (default is 1).
4. Evaluate the trained model on the test set and print the final accuracy.








