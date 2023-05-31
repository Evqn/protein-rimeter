import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim

def train_model(train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    # Check if GPU is available and if not, fall back on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Replace the last fully connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_loader.dataset.classes))

    # Move model to the device (GPU or CPU)
    model = model.to(device)

    # Define a loss function
    criterion = nn.CrossEntropyLoss()

    # Define an optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train the model
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
                dataloader = train_loader
                dataset_size = len(train_loader.sampler)
            else:
                model.eval() 
                dataloader = val_loader
                dataset_size = len(val_loader.sampler)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        torch.save(model.state_dict(), f'/Projects/protein-rimeter/models/resnet_{epoch}.pth')

    return model
