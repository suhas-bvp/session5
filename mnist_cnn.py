import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define the CNN Model
class SimpleCNN(nn.Module):
    '''def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # 1 input channel (grayscale), 32 output channels
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (28x28 -> 14x14)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (14x14 -> 7x7)

        # Fully connected layers
        # After two pooling layers, image size is 7x7. 64 channels.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(128, 10) # 10 output classes for MNIST (digits 0-9)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor for the fully connected layer
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x'''

    '''def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) # Reduced from 32 to 8 filters
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (28x28 -> 14x14)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # Reduced from 64 to 16 filters, input from conv1
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (14x14 -> 7x7)

        # Fully connected layers
        # After two pooling layers, image size is 7x7. 16 channels.
        self.fc1 = nn.Linear(16 * 7 * 7, 32) # Reduced from 64*7*7 to 16*7*7 input, and 128 to 32 output features
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(32, 10) # Input from fc1 (32), 10 output classes for MNIST (digits 0-9)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7) # Flatten the tensor for the fully connected layer, adjusted for 16 channels
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x'''
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1) # Reduced from 32 to 8 filters
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (28x28 -> 14x14)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1) # Reduced from 64 to 16 filters, input from conv1
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces image size by half (14x14 -> 7x7)

        # Fully connected layers
        # After two pooling layers, image size is 7x7. 16 channels.
        self.fc1 = nn.Linear(16 * 7 * 7, 28) # Reduced output features from 32 to 28
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(28, 10) # Input from fc1 (28), 10 output classes for MNIST (digits 0-9)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7) # Flatten the tensor for the fully connected layer, adjusted for 16 channels
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# 2. Load and Prepare Data
def get_mnist_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert image to PyTorch Tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize pixel values
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 3. Training Function
def train_model(model, train_loader, optimizer, criterion, device, epochs=5):
    model.train() # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() # Clear gradients
            output = model(data) # Forward pass
            loss = criterion(output, target) # Calculate loss
            loss.backward() # Backward pass (compute gradients)
            optimizer.step() # Update weights

            running_loss += loss.item()
            if batch_idx % 100 == 0: # Print loss every 100 batches
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        print(f'Epoch {epoch+1} finished. Average Loss: {running_loss/len(train_loader):.4f}')

# 4. Evaluation Function
def evaluate_model(model, test_loader, device):
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1) # Get the index of the max log-probability
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

# 5. Parameter Count Function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Main Execution Block
if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 1

    # Get data loaders
    train_loader, test_loader = get_mnist_dataloaders(BATCH_SIZE)

    # Initialize model
    model = SimpleCNN().to(device)

    # Print parameter count
    param_count = count_parameters(model)
    print(f"Total trainable parameters: {param_count}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, optimizer, criterion, device, EPOCHS)
    print("Training complete.")

    # Evaluate the model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {accuracy:.2f}%")
