from typing import List
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F

"""
Plot a bar chart showing the distribution of classes in train and test sets.

Params:
    - y_train: List of training labels
    - y_test: List of test labels
    - classes: List of class names
"""

def plot_distribution(y_train: List[int], 
                      y_test: List[int], 
                      classes: List[str]):
    
    train_counts = [y_train.count(i) for i in range(len(classes))]
    test_counts = [y_test.count(i) for i in range(len(classes))]

    x = np.arange(len(classes))
    bar_width = 0.35

    plt.bar(x - bar_width/2, train_counts, width=bar_width, label='Train')
    plt.bar(x + bar_width/2, test_counts, width=bar_width, label='Test')

    plt.xticks(x, classes)
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Classes in Train and Test Sets')
    plt.show()

"""
Compute the output dimensions of a convolutional layer.

Params:
    - conv_layer: Convolutional layer to compute dimensions for
    - h_in: Input height
    - w_in: Input width

Returns:
    Output height and width after convolution
"""

def out_dimensions(conv_layer: torch.nn.Conv2d, 
                   h_in: int, 
                   w_in: int):
    from math import floor

    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out


"""
Convert a PyTorch tensor to a NumPy array for image display.

Params:
    - tensor: Input tensor to convert

Returns:
    NumPy array with rearranged dimensions for matplotlib
"""

def to_numpy_imshow(tensor: torch.Tensor):
    npimg = tensor.cpu().clone().detach().numpy()
    return np.transpose(npimg, (1, 2, 0))  # Rearrange dimensions for matplotlib

"""
Train a neural network model and track training and validation losses.

Params:
    - model: Neural network model to train
    - n_epochs: Number of training epochs
    - train_loader: DataLoader for training data
    - validation_loader: DataLoader for validation data
    - test_loader: DataLoader for test data
    - optimizer: Optimizer
    - loss_fn: Loss function
    - device: Device to run training on (cpu/cuda/mps)
    - print_every_n_batches: Print loss every N batches (optional)

Returns:
    Lists of training and validation losses for each epoch
"""

def train_model(model: torch.nn.Module, 
                n_epochs: int, 
                train_loader: torch.utils.data.DataLoader,
                validation_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                device: torch.device,
                print_every_n_batches: int = None):  # New parameter
    train_loss_list = []
    validation_loss_list = []
    
    def calculate_accuracy(loader):
        n_correct = 0
        n_samples = 0
        # Set the model in evaluation mode
        model.eval()
        with torch.no_grad():
            # Iterate over the loader to get data and targets
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                # Get the predicted class by taking the index of the maximum value
                _, predicted = torch.max(outputs.data, 1)
                # Update the number of correct predictions and the number of samples
                n_samples += target.size(0)
                n_correct += (predicted == target).sum().item()
        return 100.0 * n_correct / n_samples
    
    def calculate_validation_loss():
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # Compute the loss function
                val_loss += loss_fn(output, target).item()
        return val_loss / len(validation_loader)

    for epoch in range(n_epochs):
        loss_train = 0
        batch_count = 0
        
        for data, target in train_loader:
            # Set the model in training mode
            model.train()
            data, target = data.to(device), target.to(device)
            # Set the gradient to 0
            optimizer.zero_grad()
            # Make a prediction
            output = model(data)
            # Compute the loss function
            loss = loss_fn(output, target)
            loss_train += loss.item()
            # Backpropagation
            loss.backward()
            # Update parameters
            optimizer.step()
            
            batch_count += 1
            
            # Print every N batches if specified
            if print_every_n_batches and batch_count % print_every_n_batches == 0:
                current_train_loss = loss_train / batch_count
                validation_loss = calculate_validation_loss()
                train_acc = calculate_accuracy(train_loader)
                val_acc = calculate_accuracy(validation_loader)
                print(f"Epoch {epoch + 1}, Batch {batch_count}:")
                print(f"Train loss: {current_train_loss:.4f}, Validation loss: {validation_loss:.4f}")
                print(f"Train accuracy: {train_acc:.2f}%, Validation accuracy: {val_acc:.2f}%")
                
        # Calculate average training loss for the epoch
        loss_train = loss_train / len(train_loader)
        train_loss_list.append(loss_train)
        
        # Calculate validation loss
        validation_loss = calculate_validation_loss()
        validation_loss_list.append(validation_loss)
        
        # Print epoch results if not printing batch-wise
        if not print_every_n_batches:
            train_acc = calculate_accuracy(train_loader)
            val_acc = calculate_accuracy(validation_loader)
            print(f"Epoch {epoch + 1}:")
            print(f"Train loss: {loss_train:.4f}, Validation loss: {validation_loss:.4f}")
            print(f"Train accuracy: {train_acc:.2f}%, Validation accuracy: {val_acc:.2f}%")
            
        # Calculate and print test accuracy at the end of each epoch
        test_acc = calculate_accuracy(test_loader)
        print(f"Test accuracy: {test_acc:.2f}%")
        
    return train_loss_list, validation_loss_list

"""
Plot the training and validation loss values.

Params:
    - train_loss: List of training loss values
    - validation_loss: List of validation loss values
    - epochs: Number of epochs
    - title: Plot title
"""

def plot_loss(train_loss: List[float], 
              validation_loss: List[float], 
              epochs: int, 
              title: str):
    plt.figure()
    plt.plot(range(epochs), train_loss)
    plt.plot(range(epochs), validation_loss)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.title(title)
    plt.show()

# Define the Simple_CNN model

class Simple_CNN(nn.Module):
    def __init__(self):
        super(Simple_CNN, self).__init__()

        # Initial dimensions of the images
        h_in, w_in = 32, 32

        # First convolutional block: Conv - Conv - Activ - Pool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        h_in, w_in = out_dimensions(self.conv1, h_in, w_in)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        h_in, w_in = out_dimensions(self.conv2, h_in, w_in)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Second convolutional block: Conv - Conv - Activ - Pool
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        h_in, w_in = out_dimensions(self.conv3, h_in, w_in)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        h_in, w_in = out_dimensions(self.conv4, h_in, w_in)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        final_channels = 512

        # 3 Fully connected layers
        self.fc1 = nn.Linear(final_channels * h_in * w_in, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

        # Store final dimensions
        self.dimensions_final = (final_channels, h_in, w_in)

    def forward(self, x):
        # First convolutional block: Conv - Conv - Activ - Pool
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.pool1(x)

        # Second convolutional block: Conv - Conv - Activ - Pool
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.gelu(x)

        x = self.pool2(x)

        n_channels, h, w = self.dimensions_final
        x = x.view(-1, n_channels * h * w)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
# Define the Upgraded_CNN model

class Upgraded_CNN(nn.Module):
    def __init__(self):
        super(Upgraded_CNN, self).__init__()

        # Initial dimensions of the images
        h_in, w_in = 32, 32

        # First convolutional block: Conv - BatchNorm - Activ - Pool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv1, h_in, w_in)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv2, h_in, w_in)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Second convolutional block: Conv - BatchNorm - Activ - Pool
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv3, h_in, w_in)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv4, h_in, w_in)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Third convolutional block: Conv - BatchNorm - Activ - Pool
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv5, h_in, w_in)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv6, h_in, w_in)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        final_channels = 512

        # 2 Fully connected layers: FC - BatchNorm - Activ - Dropout
        self.fc1 = nn.Linear(final_channels * h_in * w_in, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn8 = nn.BatchNorm1d(64)

        # Output layer (Fully connected layer)
        self.fc3 = nn.Linear(64, 10)

        # Dropout layer definition for regularization
        self.dropout = nn.Dropout(0.55)

        # Store final dimensions
        self.dimensions_final = (final_channels, h_in, w_in)

    def forward(self, x):
        # First convolutional block: Conv - BatchNorm - Activ - Conv - BatchNorm - Activ - Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.pool1(x)

        # Second convolutional block: Conv - BatchNorm - Activ - Conv - BatchNorm - Activ - Pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.pool2(x)

        # Third convolutional block: Conv - BatchNorm - Activ - Conv - BatchNorm - Activ - Pool
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.gelu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.gelu(x)
        x = self.pool3(x)

        n_channels, h, w = self.dimensions_final
        # Flatten the output for the fully connected layers
        x = x.view(-1, n_channels * h * w)

        # 2 Fully connected layers: FC - BatchNorm - Activ - Dropout
        x = self.fc1(x)
        x = self.bn7(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn8(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Output layer
        x = self.fc3(x)

        return x

if __name__ == '__main__':

    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    
    '''
    Q4 - Code
    '''
    # Define the transformations for the images, with normalization only and with augmentation (for the second model)
    transform_normalize_only = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])

    transform_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])

    # Load the CIFAR-10 dataset and create the loaders, for the first model
    batch_size = 32

    trainset_first_model = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_normalize_only)
    train_loader_first_model = torch.utils.data.DataLoader(trainset_first_model, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_normalize_only)

    '''
    Q5 - Code
    '''
    # Split the test set into validation and test sets
    dataset_validation, dataset_test = torch.utils.data.random_split(testset, [0.5, 0.5])

    validation_loader_first_model = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader_first_model = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    '''
    Q2 - Code
    '''
    # Get one batch of images and labels from the train loader for creating the plot
    data_iterable = iter(train_loader_first_model)
    images, labels = next(data_iterable)

    # Save one image per class
    unique_class_images = {}
    num_images = len(images)

    # Iterate over the images for getting one image per class
    for i in range(num_images):
        label = labels[i].item()
        if label not in unique_class_images:
            unique_class_images[label] = images[i]
        
        # Break if we have one image for each class
        if len(unique_class_images) == len(classes):
            break

    # Create the plot
    fig = plt.figure(figsize=(12, 4))
    plt.title("One image per class sample")
    idx = 1

    # Iterate over the unique images found
    for label in unique_class_images:
        img = unique_class_images[label]
        
        # Create subplot
        ax = fig.add_subplot(1, len(unique_class_images), idx, xticks=[], yticks=[])
        plt.imshow(to_numpy_imshow(img))
        ax.set_title(classes[label])
        
        idx += 1

    
    plt.show()

    # Plot the distribution of classes in the train and test sets
    plot_distribution(trainset_first_model.targets, testset.targets, classes)

    '''
    Q3 - Code
    '''

    # Create a dataset without transformations
    dataset_train_not_tranformed = datasets.CIFAR10('.', train=True, download=True)

    # Check the type of the first entry without transformation
    entry = dataset_train_not_tranformed[0]
    print("Entry type:", type(entry[0]))

    # Check the type of the first entry after transformation
    entry = trainset_first_model[0]
    print("Entry type after transformation:", type(entry[0]), "Tensor shape:", entry[0].shape, "Class:", entry[1])

    '''
    Q6 - Code
    '''
    # Initialize the model, optimizer and loss function
    simple_model = Simple_CNN()
    simple_learning_rate = 0.032
    first_model_epochs = 4

    simple_optimizer = optim.SGD(simple_model.parameters(), lr=simple_learning_rate)
    simple_loss_fn = nn.CrossEntropyLoss()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps'
        if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training on device {DEVICE}")
    
    simple_model = simple_model.to(DEVICE)

    '''
    Q7 - Code
    '''
    # Train the model
    print("Training the simple model")
    train_loss, validation_loss = train_model(simple_model, first_model_epochs, train_loader_first_model, validation_loader_first_model, test_loader_first_model, simple_optimizer, simple_loss_fn, DEVICE, 500)

    '''
    Q8 - Code
    '''
    # Plot the loss values
    plot_loss(train_loss, validation_loss, first_model_epochs, "Simple CNN Model Loss")
    

    '''
    Q9 - Code
    '''
    # Set again the seed for reproducibility, due to some issue during my tests
    seed = 42
    torch.manual_seed(seed)
    
    # Create the loaders for the second model, using the augmented transformation
    batch_size = 32

    trainset_second_model = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
    train_loader_second_model = torch.utils.data.DataLoader(trainset_second_model, batch_size=batch_size, shuffle=True, num_workers=2)

    validation_loader_second_model = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader_second_model = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the model, optimizer and loss function
    model = Upgraded_CNN()
    learning_rate = 0.03
    second_model_epochs = 8

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0003)
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(DEVICE)

    # Train the model
    print("Training the upgraded model")
    train_loss, validation_loss = train_model(model, second_model_epochs, train_loader_second_model, validation_loader_second_model, test_loader_second_model, optimizer, loss_fn, DEVICE)

    # Plot the loss values
    plot_loss(train_loss, validation_loss, second_model_epochs, "Upgraded CNN Model Loss")
    
    '''
    Q10 - Code
    '''
    # Iterate over different seeds
    for seed in range(5,10):
        torch.manual_seed(seed)
        print("Seed equal to ", torch.random.initial_seed())
        # To avoid transfer learning, I reinitialize the model every seed
        simple_model = Simple_CNN()
        simple_model = simple_model.to(DEVICE)
        simple_optimizer = optim.SGD(simple_model.parameters(), lr=simple_learning_rate)
        train_model(simple_model, first_model_epochs, train_loader_first_model, validation_loader_first_model, test_loader_first_model, simple_optimizer, simple_loss_fn, DEVICE)