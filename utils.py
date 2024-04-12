import torch
import albumentations
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchview import draw_graph
from torch_lr_finder import LRFinder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_inv_transforms():
    """Method to get transform to inverse the effect of normalization for ploting

    Returns:
        _Object: Object to apply image augmentations
    """
    # Normalize image
    inv_transforms = albumentations.Normalize([-0.48215841/0.24348513, -0.44653091/0.26158784, -0.49139968/0.24703223],
                                              [1/0.24348513, 1/0.26158784, 1/0.24703223], max_pixel_value=1.0)
    return inv_transforms

def plot_samples(train_loader):
    """Method to plot samples of augmented images

    Args:
        train_loader (Object): Object of data loader class to get images
    """
    inv_transform = get_inv_transforms()

    figure = plt.figure(figsize=(20,20))
    num_of_images = 10
    images, labels = next(iter(train_loader))

    for index in range(1, num_of_images + 1):
        plt.subplot(5, 5, index)
        plt.title(CLASS_NAMES[labels[index].numpy()])
        plt.axis('off')
        image = np.array(images[index])
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)

def save_samples(train_loader, path):
    """Method to plot samples of augmented images

    Args:
        train_loader (Object): Object of data loader class to get images
        path (str): Path to store plots
    """
    inv_transform = get_inv_transforms()

    figure = plt.figure(figsize=(20,8))
    num_of_images = 10
    images, labels = next(iter(train_loader))

    for index in range(1, num_of_images + 1):
        plt.subplot(2, 5, index)
        plt.title(CLASS_NAMES[labels[index].numpy()])
        plt.axis('off')
        image = np.array(images[index])
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)
    plt.savefig(path)
    plt.close(figure)

def save_model_architecture(model, filename, directory):
    """Method to save model architecture

    Args:
        model (Object): Object of model class
        filename (str): Name of image
        directory (str): Name of folder to save image
    """

    model_graph = draw_graph(model, input_size=(1,3,32,32), expand_nested=True, save_graph=True,
                             filename=filename, directory=directory)

def get_optimizer(model, lr, momentum=0, weight_decay=0, optimizer_type='SGD'):
    """Method to get object of stochastic gradient descent. Used to update weights.

    Args:
        model (Object): Neural Network model
        lr (float): Value of learning rate
        momentum (float): Value of momentum
        weight_decay (float): Value of weight decay
        optimizer_type (str): Type of optimizer SGD or ADAM

    Returns:
        object: Object of optimizer class to update weights
    """
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def get_StepLR_scheduler(optimizer, step_size, gamma):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        step_size (int): Period of learning rate decay
        gamma (float): Number to multiply with learning rate

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
    return scheduler

def get_ReduceLROnPlateau_scheduler(optimizer, factor, patience):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        factor (float): Number to multiply with learning rate
        patience (int): Number of epoch to wait

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)
    return scheduler

def get_OneCycleLR_scheduler(optimizer, max_lr, epochs, steps_per_epoch, max_at_epoch, anneal_strategy, div_factor, final_div_factor):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        max_lr (float): Maximum learning rate to reach during training
        epochs (float): Total number of epoch
        steps_per_epoch (int): Total steps in an epoch
        max_at_epoch (int): Epoch to reach maximum learning rate
        anneal_strategy (string): Strategy to interpolate between minimum and maximum lr
        div_factor (int): Divisive factor to calculate intial learning rate
        final_div_factor (int): Divisive factor to calculate minimum learning rate

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,  epochs=epochs,
                                              steps_per_epoch=steps_per_epoch, 
                                              pct_start=max_at_epoch/epochs,
                                              anneal_strategy=anneal_strategy,
                                              div_factor=div_factor,
                                              final_div_factor=final_div_factor)
    return scheduler

def get_criterion(loss_type='cross_entropy'):
    """Method to get loss calculation ctiterion

    Args:
        loss_type (str): Type of loss 'nll_loss' or 'cross_entropy' loss

    Returns:
        object: Object to calculate loss 
    """
    if loss_type == 'nll_loss':
        criterion = F.nll_loss
    elif loss_type == 'cross_entropy':
        criterion = F.cross_entropy
    return criterion

def get_learning_rate(model, optimizer, criterion, device, trainloader):
    """Method to find learning rate using LR finder.

    Args:
        model (Object): Object of model
        optimizer (Object): Object of optimizer class
        criterion (Object): Loss function
        device (string): Type of device "cuda" or "cpu"
        trainloader (Object): Object of dataloader class

    Returns:
        float: Learning rate suggested by lr finder
    """
    # Create object and perform range test
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainloader, end_lr=100, num_iter=100)

    # Plot result and store suggested lr
    plot, suggested_lr = lr_finder.plot()

    # Reset model and optimizer
    lr_finder.reset()

    return suggested_lr

def plot_accuracy_loss_graphs(train_losses, train_acc, test_losses, test_acc):
    """Method to plot loss and accuracy of training and testing

    Args:
        train_losses (List): List containing loss of model after each epoch on training data
        train_acc (List): List containing accuracy of model after each epoch on training data
        test_losses (List): List containing loss of model after each epoch on testing data
        test_acc (List): List containing accuracy of model after each epoch on testing data
    """
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    # Plot training losses
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    # Plot training accuracy
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    # Plot test losses
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    # Plot test aacuracy
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    

def save_accuracy_loss_graphs(train_losses, train_acc, test_losses, test_acc, path):
    """Method to plot loss and accuracy of training and testing

    Args:
        train_losses (List): List containing loss of model after each epoch on training data
        train_acc (List): List containing accuracy of model after each epoch on training data
        test_losses (List): List containing loss of model after each epoch on testing data
        test_acc (List): List containing accuracy of model after each epoch on testing data
        path (str): Path to save accuracy and loss plots
    """
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    # Plot training losses
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    # Plot training accuracy
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    # Plot test losses
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    # Plot test aacuracy
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    fig.savefig(path)
    plt.close(fig)

def plot_missclassified_images(device, model, test_loader):
    """Method to plot missclassified images

    Args:
        device (string): Type of device "cuda" or "cpu"
        model (Object): Object of model
        test_loader (Object): Object of dataloader class
    """
    model.eval()
    inv_transform = get_inv_transforms()
    missclassified_image_list = []
    label_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            if len(missclassified_image_list) > 10:
                break
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    missclassified_image_list.append(data[i])
                    label_list.append(CLASS_NAMES[target[i]])
                    pred_list.append(CLASS_NAMES[pred[i]])

    figure = plt.figure(figsize=(20,20))
    num_of_images = 10
    for index in range(1, num_of_images + 1):
        plt.subplot(5, 5, index)
        plt.title(f'Actual: {label_list[index]} Prediction: {pred_list[index]}')
        plt.axis('off')
        image = np.array(missclassified_image_list[index].cpu())
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)

def save_missclassified_images(device, model, test_loader, path):
    """Method to plot missclassified images

    Args:
        device (string): Type of device "cuda" or "cpu"
        model (Object): Object of model
        test_loader (Object): Object of dataloader class
        path (string): Path to save missclassified images
    """
    model.eval()
    inv_transform = get_inv_transforms()
    missclassified_image_list = []
    label_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            if len(missclassified_image_list) > 10:
                break
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    missclassified_image_list.append(data[i])
                    label_list.append(CLASS_NAMES[target[i]])
                    pred_list.append(CLASS_NAMES[pred[i]])

    figure = plt.figure(figsize=(20,8))
    num_of_images = 10
    for index in range(1, num_of_images + 1):
        plt.subplot(2, 5, index)
        plt.title(f'Actual: {label_list[index]} Prediction: {pred_list[index]}')
        plt.axis('off')
        image = np.array(missclassified_image_list[index].cpu())
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)
    plt.savefig(path)
    plt.close(figure)

def save_grad_cam_images(device, model, test_loader, path, target_layers):
    """Method to plot missclassified images

    Args:
        device (string): Type of device "cuda" or "cpu"
        model (Object): Object of model
        test_loader (Object): Object of dataloader class
        path: (string): path to save grad-CAM results
        target_layer (Object): Convolution layer to extract the feature maps
    """
    model.eval()
    inv_transform = get_inv_transforms()
    missclassified_image_list = []
    label_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            if len(missclassified_image_list) > 10:
                break
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    missclassified_image_list.append(data[i])
                    label_list.append(CLASS_NAMES[target[i]])
                    pred_list.append(CLASS_NAMES[pred[i]])

    cam = GradCAM(model=model, target_layers=target_layers)

    figure = plt.figure(figsize=(20,8))
    num_of_images = 10
    for index in range(1, num_of_images + 1):
        plt.subplot(2, 5, index)
        plt.title(f'Actual: {label_list[index]} Prediction: {pred_list[index]}')
        plt.axis('off')
        input_tensor = missclassified_image_list[index].cpu()
        targets = [ClassifierOutputTarget(CLASS_NAMES.index(pred_list[index]))]

        grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)

        grayscale_cam = grayscale_cam[0, :]
        image = np.array(missclassified_image_list[index].cpu())
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        image = np.clip(image, 0, 1)
        visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        plt.imshow(visualization)
    plt.savefig(path)
    plt.close(figure)
