# Dl_Mini_project
This project explores the task of image classification on the CIFAR-10 dataset using a custom ResNet architec- ture


This code trains a model to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes. The architecture is based on ResNet and includes residual blocks with squeeze-and-excitation (SE) blocks for improved feature representation. 

Key Features:
- Data augmentation techniques such as random cropping, horizontal flipping, rotation, affine transformations, and color jittering are applied to improve model generalization.
- Stochastic Gradient Descent (SGD) optimizer with a moderate learning rate, weight decay, and gradient clipping is used for training.
- The learning rate is adjusted using a cosine annealing scheduler to improve convergence.
- The model is trained for 200 epochs, with early stopping based on validation accuracy.
- The best model is saved and evaluated on the test dataset to obtain the final accuracy.

To run the code, ensure you have PyTorch, torchvision, and torchsummary installed. Then, simply execute the provided script. The training process will display the loss and accuracy metrics for each epoch, and the final test accuracy will be reported."
