# Convolutional Neural Network for CIFAR-10 Image Classification
Convolutional Neural Network for image classification using the CIFAR-10 dataset.


The key components and achievements of the project are summarized below:

## Data Exploration and Preprocessing

1. Loaded the CIFAR-10 dataset using `torchvision.datasets.CIFAR10` and visualized one image per class. Ensured a balanced distribution of images in both training and test sets.

2. Transformed the dataset using `transforms.ToTensor()` to convert image elements into suitable types and normalized the images to have a mean of 0 and a standard deviation of 1 using `transforms.Normalize`.

3. Split the original training set into new training (80%) and validation (20%) sets using `torch.utils.data.random_split`.

## Convolutional Neural Network Architecture

Implemented a CNN with the following architecture:

- Four convolutional layers (conv1 to conv4) with ReLU activation functions.
- Max pooling layers for spatial downsampling after each pair of convolutional layers.
- Fully connected layer (fc1) producing an output vector of size 10 for the ten classes in CIFAR-10.

## Training

1. Developed a training pipeline with a loop iterating through epochs and nested loops for batches of the training dataset. Utilized cross-entropy loss and stochastic gradient descent (SGD) with momentum for optimization.

2. Validated the model's performance using a separate validation set. Printed and recorded the current validation loss and accuracy per epoch for hyperparameter tuning.

3. Achieved a test accuracy of 75.36% with a batch size of 32 and 4 epochs.

4. Saved the parameters of the trained model using `torch.save(net.state_dict(), 'MATTIA_COLBERTALDO_1.pt')`.

5. Visualized the evolution of both training and validation losses and accuracies over epochs.

6. Improved the model, reaching a test accuracy of 83%, and saved the new model parameters using `torch.save(model.state_dict(), 'MATTIA_COLBERTALDO_2.pt')`.

## Bonus

Averaged the fastest model on 10 different seeds, resulting in a mean accuracy of 75.91% with a standard deviation of 0.34%. Evaluated model performance and robustness based on mean accuracy and standard deviation across multiple runs.


Feel free to explore the provided visualizations and code for more details on the project!
