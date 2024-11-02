from typing import List, Tuple
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F
from math import floor

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
    plt.show(block=False)

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
    - optimizer: Optimization algorithm
    - loss_fn: Loss function
    - device: Device to run training on (cpu/cuda/mps)

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
                device: torch.device):
    train_loss_list = []
    validation_loss_list = []
    for epoch in range(n_epochs):
        loss_train = 0
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
        loss_train = loss_train / len(train_loader)
        train_loss_list.append(loss_train)

        # At the end of every epoch, check the validation loss value
        with torch.no_grad():
            model.eval()
            for data, target in validation_loader: # Just one batch
                data, target = data.to(device), target.to(device)
                # Make a prediction
                output = model(data)
                # Compute the loss function
                validation_loss = loss_fn(output, target).item()
        print(f"Epoch {epoch + 1}: Train loss: {loss_train}, Validation loss {validation_loss}")
        validation_loss_list.append(validation_loss)
        
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += target.size(0)
                n_correct += (predicted == target).sum().item()

            acc = 100.0 * n_correct / n_samples
        print("Accuracy on the test set:", acc, "%")
    return train_loss_list, validation_loss_list

def plot_loss(train_loss, validation_loss, epochs):
    plt.figure()
    plt.plot(range(epochs), train_loss)
    plt.plot(range(epochs), validation_loss)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.show(block=False)

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

class Upgraded_CNN(nn.Module):
    def __init__(self):
        super(Upgraded_CNN, self).__init__()

        # Initial dimensions of the images
        h_in, w_in = 32, 32

        # First convolutional block: Conv - BatchNorm - Activ - Pool - Dropout
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv1, h_in, w_in)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv2, h_in, w_in)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Second convolutional block: Conv - BatchNorm - Activ - Pool - Dropout
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv3, h_in, w_in)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv4, h_in, w_in)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Third convolutional block: Conv - BatchNorm - Activ - Pool - Dropout
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

        # Output layer
        self.fc3 = nn.Linear(64, 10)

        # Dropout layers definition for regularization
        self.weak_dropout = nn.Dropout(0.2)
        self.strong_dropout = nn.Dropout(0.4)

        # Store final dimensions
        self.dimensions_final = (final_channels, h_in, w_in)

    def forward(self, x):
        # First convolutional block: Conv - BatchNorm - Activ - Pool - Weak Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.pool1(x)
        x = self.weak_dropout(x)

        # Second convolutional block: Conv - BatchNorm - Activ - Pool - Weak Dropout
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.pool2(x)
        x = self.weak_dropout(x)

        # Third convolutional block: Conv - BatchNorm - Activ - Pool - Weak Dropout
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.gelu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.gelu(x)
        x = self.pool3(x)
        x = self.weak_dropout(x)

        n_channels, h, w = self.dimensions_final
        # Flatten the output for the fully connected layers
        x = x.view(-1, n_channels * h * w)

        # 2 Fully connected layers: FC - BatchNorm - Activ - Strong Dropout
        x = self.fc1(x)
        x = self.bn7(x)
        x = F.gelu(x)
        x = self.strong_dropout(x)
        x = self.fc2(x)
        x = self.bn8(x)
        x = F.gelu(x)
        x = self.strong_dropout(x)

        # Output layer
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    
    '''
    Q4 - Code
    '''

    batch_size = 32

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

    trainset_32 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_normalize_only)
    train_loader_32 = torch.utils.data.DataLoader(trainset_32, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_normalize_only)

    '''
    Q5 - Code
    '''
    dataset_validation, dataset_test = torch.utils.data.random_split(testset, [0.5, 0.5])

    validation_loader_32 = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader_32 = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    '''
    Q2 - Code
    '''

    data_iterable = iter(train_loader_32)
    images, labels = next(data_iterable)

    # Save one image per class
    unique_class_images = {}
    num_images = len(images)

    for i in range(num_images):
        label = labels[i].item()
        if label not in unique_class_images:
            unique_class_images[label] = images[i]
        
        # Break if we have one image for each class
        if len(unique_class_images) == len(classes):
            break

    # Plot the images
    fig = plt.figure(figsize=(12, 4))
    idx = 1

    # Plot each saved imag
    for label in unique_class_images:
        img = unique_class_images[label]
        
        # Create subplot
        ax = fig.add_subplot(1, len(unique_class_images), idx, xticks=[], yticks=[])
        plt.imshow(to_numpy_imshow(img))
        ax.set_title(classes[label])
        
        idx += 1

    plt.show(block=False)


    plot_distribution(trainset_32.targets, testset.targets, classes)

    '''
    Q3 - Code
    '''

    dataset_train_not_tranformed = datasets.CIFAR10('.', train=True, download=True)

    entry = dataset_train_not_tranformed[0]
    print("Entry type:", type(entry[0]))

    entry = trainset_32[0]
    print("Entry type after transformation:", type(entry[0]), "Tensor shape:", entry[0].shape, "Class:", entry[1])

    '''
    Q6 - Code
    '''

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

    train_loss, validation_loss = train_model(simple_model, first_model_epochs, train_loader_32, validation_loader_32, test_loader_32, simple_optimizer, simple_loss_fn, DEVICE)

    '''
    Q8 - Code
    '''
    plot_loss(train_loss, validation_loss, first_model_epochs)
    

    '''
    Q9 - Code
    '''
    seed = 42
    torch.manual_seed(seed)
    
    batch_size = 40

    trainset_40 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_augmented)
    train_loader_40 = torch.utils.data.DataLoader(trainset_40, batch_size=batch_size, shuffle=True, num_workers=2)

    validation_loader_40 = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader_40 = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)


    model = Upgraded_CNN()
    learning_rate = 0.031
    second_model_epochs = 8

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(DEVICE)

    train_loss, validation_loss = train_model(model, second_model_epochs, train_loader_40, validation_loader_40, test_loader_40, optimizer, loss_fn, DEVICE)
    plot_loss(train_loss, validation_loss, second_model_epochs)
    plt.show()
    '''
    Q10 - Code
    '''
    for seed in range(5,10):
        torch.manual_seed(seed)
        print("Seed equal to ", torch.random.initial_seed())
        
        simple_model = Simple_CNN()
        train_model(simple_model, first_model_epochs, train_loader_32, validation_loader_32, test_loader_32, simple_optimizer, simple_loss_fn, DEVICE)