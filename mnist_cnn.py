import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 1. Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), # Reduced from 32 to 8 filters
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces image size by half (28x28 -> 14x14)
            nn.Dropout2d(0.0),

            # Second convolutional layer
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # Reduced from 64 to 16 filters, input from conv1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces image size by half (14x14 -> 7x7)
            nn.Dropout2d(0.1),

            # Third convolutional layer
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # Reduced from 64*4 to 128 filters, input from conv1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces image size by half (7x7 -> 3x3)
            nn.Dropout2d(0.2),   
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 3, 10) # Reduced output features from 32 to 10
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,32 * 3 * 3)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

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
        total_samples=0.0
        correct_predictions=0.0
        epoch_accuracy=0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad() # Clear gradients
            output = model(data) # Forward pass
            loss = criterion(output, target) # Calculate loss
            loss.backward() # Backward pass (compute gradients)
            optimizer.step() # Update weights
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1) # Get the class with the highest score
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            epoch_accuracy = (correct_predictions / total_samples) * 100

            #if batch_idx % 100 == 0: # Print loss every 100 batches
            #    print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
        
        print(f'Epoch {epoch+1} finished. Average Loss: {running_loss/len(train_loader):.4f} && Training Accuracy: {epoch_accuracy:.2f}%')
        #print(f'Epoch {epoch+1} finished. Training Accuracy: {epoch_accuracy:.2f}%')
        
       
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

    #print(f'Accuracy on test set: {accuracy:.2f}%')
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
    LEARNING_RATE = 0.0005
    EPOCHS = 15

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
