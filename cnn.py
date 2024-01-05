# -*- coding: utf-8 -*-
'''
Assignment 2
Student: MATTIA COLBERTALDO
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim



if __name__ == '__main__':


    torch.cuda.get_arch_list()
    print(torch.cuda.get_device_name(0))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set the seed in torch
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    batch_size = 32

    dataset = torchvision.datasets.CIFAR10(root='./data', transform=transforms.ToTensor(), download=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Calculate mean and std dev across all batches
    mean_list, std_list = [], []

    for batch in dataloader:
        images, _ = batch
        mean_list.append(torch.mean(images, dim=(0, 2, 3)))
        std_list.append(torch.std(images, dim=(0, 2, 3)))

    # Calculate the overall mean and std dev
    overall_mean = torch.stack(mean_list).mean(dim=0)
    overall_std = torch.stack(std_list).mean(dim=0)

    print(f"Overall Mean: {overall_mean}")
    print(f"Overall Std: {overall_std}")

    # Let's create an instance of our network
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(overall_mean, overall_std)
    ])



    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)


    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Display one image per class
    class_indices = [train_dataset.targets.index(class_idx) for class_idx in range(len(classes))]

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    images = []
    for i in class_indices:
        img, label = train_dataset[i]
        images.append(img)
    imshow(torchvision.utils.make_grid(images))
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(len(classes))))
    print(classes)

    sample_image, _ = train_dataset[0]
    print(f"Image dimension: {sample_image.shape}")

    # Display histogram of the distribution of images in the training set
    plt.figure(figsize=(10, 5))
    plt.hist(train_dataset.targets, bins=range(11), align='left', rwidth=0.8)
    plt.xticks(range(10), classes)
    plt.title('Distribution of Images in Training Set')
    plt.show()

    # Display histogram of the distribution of images in the test set
    plt.figure(figsize=(10, 5))
    plt.hist(test_dataset.targets, bins=range(11), align='left', rwidth=0.8)
    plt.xticks(range(10), classes)
    plt.title('Distribution of Images in Test Set')
    plt.show()


    # Split the dataset into train, validation, and test sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0, pin_memory=True)



    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)



    # show batch_size images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=0, stride=1)
            self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(512, 256, kernel_size=5, padding=0, stride=1)
            self.conv4 = nn.Conv2d(256, 128, kernel_size=5, padding=0, stride=1)
            self.fc1 = nn.Linear(1152, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
            x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
            x = x.flatten(1)
            x = self.fc1(x)
            return x


    # Move the model to GPU
    net = ConvNet().cuda()

    # Define the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    # ...
    n_epochs = 4

    train_losses = []
    val_losses = []
    # Saving accuracy for future use
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()

            # Forward pass
            outputs = net(images)
            loss_train = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            running_loss += loss_train.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            accuracy_train = correct_train / total_train
            # Print statistics
            if (i + 1) % 300 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Training Accuracy: {:.2f}%".format(epoch + 1, n_epochs, i + 1, len(train_loader), loss_train.item(), accuracy_train * 100))

        train_losses.append(running_loss / len(train_loader))
        # Saving accuracy
        train_accuracies.append(accuracy_train)

        # Evaluation loop (on the entire validation set) after each epoch
        net.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for i, (images_val, labels_val) in enumerate(val_loader):
                images_val, labels_val = images_val.cuda(), labels_val.cuda()
                y_hat_val = net(images_val)
                loss_val = loss_fn(y_hat_val, labels_val)
                val_loss += loss_val.item()

                _, predicted = torch.max(y_hat_val, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted == labels_val).sum().item()

            accuracy_val = correct_val / total_val
            print("Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%".format(epoch + 1, n_epochs, loss_val.item(), accuracy_val * 100))

            val_losses.append(val_loss / len(val_loader))
            # Saving accuracy
            val_accuracies.append(correct_val / total_val)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Test the model
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Print test accuracy
    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')

torch.save(net.state_dict(), 'MATTIA_COLBERTALDO_1.pt')

import matplotlib.pyplot as plt
# Plotting training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting training and validation accuracy
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""# CNN"""

torch.cuda.get_arch_list()
print(torch.cuda.get_device_name(0))
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


batch_size = 32


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=1)
        self.activation = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=5, padding=0, stride=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=5, padding=0, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(1152, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)



    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.batchnorm1(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.activation(self.batchnorm2(x))
        x = self.pool(x)
        x = x.flatten(1)
        #print(x.shape)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


batch_size = 32

dataset = torchvision.datasets.CIFAR10(root='./data', transform=transforms.ToTensor(), download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

data = next(iter(dataloader))
images, labels = data
mean = torch.mean(images, dim=(0, 2, 3))
std = torch.std(images, dim=(0, 2, 3))

print(f"Mean: {mean}")
print(f"Std: {std}")

# Let's create an instance of our network
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=0, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0, pin_memory=True)


# Instantiate the model, loss function, and optimizer
model = CNN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)



# Training loop
num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)

    # Print validation accuracy
    val_accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy*100:.1f}%')

# Save the trained model
torch.save(model.state_dict(), 'MATTIA_COLBERTALDO_2.pt')

"""Test CNN"""

# Test the model
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

# Print test accuracy
test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy*100:.0f}%')

"""# ALL SEEDS"""

# List to store accuracy results from multiple runs
accuracies = []

# Training loop with different seeds
for seed in range(10):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #############################

    # Move the model to GPU
    net = ConvNet().cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

    # ...
    n_epochs = 4

    batch_size = 32

    train_losses = []
    val_losses = []
    # Saving accuracy for future use
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()

            # Forward pass
            outputs = net(images)
            loss_train = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            running_loss += loss_train.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        accuracy_train = correct_train / total_train
        print("Seed[{}/{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(seed, 9, epoch + 1, n_epochs, i + 1, len(train_loader), loss_train.item(), accuracy_train * 100))

        train_losses.append(running_loss / len(train_loader))
        # Saving accuracy
        train_accuracies.append(accuracy_train)

        # Evaluation loop (on the entire validation set) after each epoch
        net.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images_val, labels_val in val_loader:
                images_val, labels_val = images_val.cuda(), labels_val.cuda()
                y_hat_val = net(images_val)
                loss_val = loss_fn(y_hat_val, labels_val)
                val_loss += loss_val.item()

                _, predicted = torch.max(y_hat_val, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted == labels_val).sum().item()

            val_losses.append(val_loss / len(val_loader))
            # Saving accuracy
            val_accuracies.append(correct_val / total_val)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # Test the model
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Print test accuracy
    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')

    accuracies.append(test_accuracy)

    #############################

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print(f'Mean Accuracy: {mean_accuracy*100:.2f}%, Std Deviation: {std_accuracy*100:.2f}%')
print(accuracies)