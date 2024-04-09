import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import utils
import dataset
import argparse

from tqdm import tqdm
from models import *

def _set_seed(cuda):
    SEED = 1
    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help="Learning rate to train model")
    parser.add_argument('--test_size', default=0.2, type=float, help="Portion of data to use for testing e.g. 0.2")
    parser.add_argument('--epochs', default=20, type=int, help="Number of training epochs e.g 25")
    parser.add_argument('--batch_size', default=128, type=int, help="Number of images per batch e.g. 256")
    parser.add_argument('--optimizer', default="SGD", type=str, help="Type of optimizer e.g. ADAM")
    parser.add_argument('--scheduler', default=None, type=str, help="Type of leraning rate scheduler e.g OneCycleLR")
    parser.add_argument('--loss_type', default='cross_entropy', type=str, help='Type of loss e.g. nll_loss')
    parser.add_argument('--model_name', default="resnet18", type=str, help="Type of model to use e.g. resnet18")
    args = parser.parse_args()
    return args

def GetCorrectPredCount(pPrediction, pLabels):
    """Method to get count of correct predictions

    Args:
        pPrediction (Tensor): Tensor containing prediction from model
        pLabels (Tensor): Tensor containing true labels from data

    Returns:
        int: Number of correct predictions
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def _train(model, device, train_loader, optimizer, scheduler, criterion, train_losses, train_acc):
    """Method to train model for one epoch 

    Args:
        model (Object): Neural Network model
        device : torch.device indicating available device for training cuda/cpu
        train_loader (Object): Object of DataLoader class for training data
        optimizer (Object): Object of optimizer to update weights
        criterion (Object): To calculate loss
        train_losses (List): To store training loss
        train_acc (List): To store training accuracy
    """
    # Set model to training
    model.train()
    # Create progress bar
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    # Loop through batches of data
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device) # Store data and target to device
        optimizer.zero_grad() # Set gradients to zero

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item() # Update train loss

        # Backpropagation
        loss.backward() # Compute gradients
        optimizer.step() # Updates weights
        if scheduler != None:
            scheduler.step() # Update learning rate

        correct += GetCorrectPredCount(pred, target) # Store correct prediction count
        processed += len(data) # Store amount of data processed

        # Print results
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Append accuracy and losses to list
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def _test(model, device, test_loader, criterion, test_losses, test_acc):
    """Method to test model

    Args:
        model (Object): Neural Network model
        device : torch.device indicating available device for testing cuda/cpu
        test_loader (Object): Object of DataLoader class for testing data
        criterion (Object): To calculate loss
        test_losses (List): To store testing loss
        test_acc (List): To store testing accuracy
    """
    # Set model to eval
    model.eval()

    test_loss = 0
    correct = 0

    # Disable gradient calculation
    with torch.no_grad():
        # iterate though data and calculate loss
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) # Store data and target to device

            output = model(data) # get prediction
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target) # Store correct prediction count

    # Append accuracy and losses to list
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    # Print results
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def start_training(num_epochs, model, device, train_loader, test_loader, optimizer, criterion, scheduler):
    """Method to start training the model

    Args:
        num_epochs (int): number of epochs
        model (Object): Neural Network model
        device : torch.device indicating available device for training cuda/cpu
        train_loader (Object): Object of DataLoader class for training data
        test_loader (Object): Object of DataLoader class for testing data
        optimizer (Object): Object of optimizer to update weights
        criterion (Object): To calculate loss
        scheduler (Object): To update learning rate
    
    Returns:
        lists: Lists containing information about losses and accuracy during training and testing
    """
    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        if scheduler != None:
            # Print learning rate
            print(f'Current learning rate: {scheduler.get_last_lr()}')
        # Train for one epochs
        _train(model, device, train_loader, optimizer, scheduler, criterion, train_losses, train_acc)
        # Test model
        _test(model, device, test_loader, criterion, test_losses, test_acc)

    return train_losses, train_acc, test_losses, test_acc

def get_model(model_name, device):
    if model_name=='resnet18':
        return ResNet18().to(device)
    elif model_name=='resnet34':
        return ResNet34().to(device)

def main():
    args = get_args()

    os.makedirs('images', exist_ok=True)

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    _set_seed(cuda)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=args.batch_size, num_workers=4,
                           pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_loader = dataset.get_train_data_loader(**dataloader_args)
    test_loader = dataset.get_test_data_loader(**dataloader_args)

    utils.save_samples(train_loader, 'images/augmentation.png')

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(args.model_name, device)

    utils.save_model_architecture(model, filename="ResNet", directory="images")

    optimizer = utils.get_optimizer(model, lr=args.lr, momentum=0.9, optimizer_type=args.optimizer)
    scheduler = None
    if args.scheduler == "StepLR":
        scheduler = utils.get_StepLR_scheduler(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = utils.get_ReduceLROnPlateau_scheduler(optimizer, factor=0.1, patience=10)
    elif args.scheduler == "OneCycleLR":
        max_lr = utils.get_learning_rate(model, optimizer, criterion, device, train_loader)
        scheduler = utils.get_OneCycleLR_scheduler(optimizer, max_lr=max_lr,  epochs=args.epochs,
                                           steps_per_epoch=len(train_loader), max_at_epoch=5,
                                           anneal_strategy = 'linear', div_factor=10,
                                           final_div_factor=1)

    criterion = utils.get_criterion(loss_type=args.loss_type)

    train_losses, train_acc, test_losses, test_acc = start_training(
        args.epochs, model, device, train_loader, test_loader, optimizer, criterion,
        scheduler
    )

    utils.save_accuracy_loss_graphs(train_losses, train_acc, test_losses, test_acc, 'images/metrics.png')

    utils.save_missclassified_images(device, model, test_loader, 'images/results.png')

    utils.save_grad_cam_images(device, model, test_loader, 'images/grad_cam.png', [model.layer3[-1]])

if __name__ == "__main__":
    main()