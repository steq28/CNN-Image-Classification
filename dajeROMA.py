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

def plot_distribution(y_train, y_test, classes):
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

def out_dimensions(conv_layer, h_in, w_in):
    '''
    This function computes the output dimension of each convolutional layers in the most general way.
    '''
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Initial dimensions for CIFAR-10
        h_in, w_in = 32, 32

        # First block: Conv - Conv - Activ - Pool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        h_in, w_in = out_dimensions(self.conv1, h_in, w_in)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        h_in, w_in = out_dimensions(self.conv2, h_in, w_in)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Second block: Conv - Conv - Activ - Pool
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        h_in, w_in = out_dimensions(self.conv3, h_in, w_in)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        h_in, w_in = out_dimensions(self.conv4, h_in, w_in)

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Store final dimensions
        self.final_h = h_in
        self.final_w = w_in
        self.final_channels = 512

        # Fully connected layer
        self.fc1 = nn.Linear(self.final_channels * self.final_h * self.final_w, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

        self.dimensions_final = (self.final_channels, h_in, w_in)

    def forward(self, x):
        # First block: Conv - Conv - Activ - Pool
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.pool1(x)

        # Second block: Conv - Conv - Activ - Pool
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
    
def train_model(model, n_epochs):
    train_loss_list = []
    validation_loss_list = []
    for epoch in range(n_epochs):
        loss_train = 0
        for data, target in train_loader:
            # Set the model in training mode
            model.train()
            data, target = data.to(DEVICE), target.to(DEVICE)
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
        loss_train = loss_train / len(train_loader) # Consider this alternative method of tracking training loss.
        train_loss_list.append(loss_train)

        # At the end of every epoch, check the validation loss value
        with torch.no_grad():
            model.eval()
            for data, target in validation_loader: # Just one batch
                data, target = data.to(DEVICE), target.to(DEVICE)
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
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += target.size(0)
                n_correct += (predicted == target).sum().item()

            acc = 100.0 * n_correct / n_samples
        print("Accuracy on the test set:", acc, "%")
    return train_loss_list, validation_loss_list

class CNNSuper(nn.Module):
    def __init__(self):
        super(CNNSuper, self).__init__()
        # Initial dimensions for CIFAR-10
        h_in, w_in = 32, 32

        # First block: Conv - BatchNorm - Activ - Pool - Dropout
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv1, h_in, w_in)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv2, h_in, w_in)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Second block: Conv - BatchNorm - Activ - Pool - Dropout
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv3, h_in, w_in)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv4, h_in, w_in)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Third block: Conv - BatchNorm - Activ - Pool - Dropout
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv5, h_in, w_in)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        h_in, w_in = out_dimensions(self.conv6, h_in, w_in)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        h_in, w_in = h_in // 2, w_in // 2

        # Store final dimensions
        self.final_h = h_in
        self.final_w = w_in
        self.final_channels = 512

        # Fully connected layers
        self.fc1 = nn.Linear(self.final_channels * self.final_h * self.final_w, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)

        self.weak_dropout = nn.Dropout(0.2)
        self.strong_dropout = nn.Dropout(0.35)

    def forward(self, x):
        # First block: Conv - BatchNorm - Activ - Pool - Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.pool1(x)
        x = self.weak_dropout(x)

        # Second block: Conv - BatchNorm - Activ - Pool - Dropout
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.pool2(x)
        x = self.weak_dropout(x)

        # Third block: Conv - BatchNorm - Activ - Pool - Dropout
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.gelu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.gelu(x)
        x = self.pool3(x)
        x = self.weak_dropout(x)

        x = x.view(-1, self.final_channels * self.final_h * self.final_w)
        x = self.fc1(x)
        x = self.bn7(x)
        x = F.gelu(x)
        x = self.strong_dropout(x)
        x = self.fc2(x)
        x = self.bn8(x)
        x = F.gelu(x)
        x = self.strong_dropout(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    manual_seed = 42
    torch.manual_seed(manual_seed)
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=0, std=1),
    # ])

    # batch_size = 32

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # dataset_validation, dataset_test = torch.utils.data.random_split(testset, [0.5, 0.5])
    # validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # def to_numpy_imshow(tensor):
    #     # Convert tensor to numpy image format
    #     npimg = tensor.cpu().clone().detach().numpy()
    #     return np.transpose(npimg, (1, 2, 0))  # Rearrange dimensions for matplotlib

    # data_iterable = iter(train_loader)
    # images, labels = next(data_iterable)

    # # Save one image per class
    # unique_class_images = {}
    # num_images = len(images)

    # for i in range(num_images):
    #     label = labels[i].item()
    #     if label not in unique_class_images:
    #         unique_class_images[label] = images[i]
        
    #     # Break if we have one image for each class
    #     if len(unique_class_images) == len(classes):
    #         break

    # # Plot the images
    # fig = plt.figure(figsize=(12, 4))
    # idx = 1

    # # Plot each saved imag
    # for label in unique_class_images:
    #     img = unique_class_images[label]
        
    #     # Create subplot
    #     ax = fig.add_subplot(1, len(unique_class_images), idx, xticks=[], yticks=[])
    #     plt.imshow(to_numpy_imshow(img))
    #     ax.set_title(classes[label])
        
    #     idx += 1

    # plt.show()


    # plot_distribution(trainset.targets, testset.targets, classes)

    # dataset_train_not_tranformed = datasets.CIFAR10('.', train=True, download=True)

    # entry = dataset_train_not_tranformed[0]
    # print("Entry type:", type(entry[0]))

    # entry = trainset[0]
    # print("Entry type after transformation:", type(entry[0]), "Tensor shape:", entry[0].shape, "Class:", entry[1])

    # model = CNN()
    # learning_rate = 0.031
    # first_model_epochs = 4

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # loss_fn = nn.CrossEntropyLoss()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps'
        if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training on device {DEVICE}")
    # model = model.to(DEVICE)

    # train_loss, validation_loss = train_model(model, first_model_epochs)

    # plt.figure()
    # plt.plot(range(first_model_epochs), train_loss)
    # plt.plot(range(first_model_epochs), validation_loss)
    # plt.legend(["Train loss", "Validation Loss"])
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss value")
    # plt.show()


    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])

    batch_size = 40

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_validation, dataset_test = torch.utils.data.random_split(testset, [0.5, 0.5])
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = CNNSuper()
    learning_rate = 0.031
    second_model_epochs = 6

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(DEVICE)

    train_loss, validation_loss = train_model(model, second_model_epochs)

    plt.figure()
    plt.plot(range(second_model_epochs), train_loss)
    plt.plot(range(second_model_epochs), validation_loss)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.show()