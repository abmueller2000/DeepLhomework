from threading import current_thread
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from os import path
from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import warnings

warnings.filterwarnings("ignore",
                        "The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*")


def train(args):
    # Initialize the model and move it to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCN().to(device)

    # Convert DENSE_CLASS_DISTRIBUTION to a tensor and move to the same device as the model
    class_weights = torch.tensor(DENSE_CLASS_DISTRIBUTION, dtype=torch.float32).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.10)

    # Define the transformations
    transformations = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ColorJitter(brightness=1.3, contrast=1.3, saturation=1.3, hue=0.5),
        dense_transforms.ToTensor(),
        dense_transforms.Normalize(mean=[0.3321, 0.3219, 0.3267], std=[0.2554, 0.2318, 0.2434])
    ])

    # Initialize the loaders
    train_loader = load_dense_data("dense_data/train", args.num_workers, args.batch_size, transform=transformations)
    valid_loader = load_dense_data("dense_data/valid", args.num_workers, args.batch_size)

    # Initialize the loggers with a default directory if not provided
    log_dir = args.log_dir if args.log_dir else 'logs'
    train_logger = SummaryWriter(path.join(log_dir, 'train'))
    valid_logger = SummaryWriter(path.join(log_dir, 'valid'))

    top_valid_IOU = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        current_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader, 0):
            # Move data and target to the same device as the model
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            target = target.to(torch.int64)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            _, prediction = torch.max(output.data, 1)
            accuracy = (prediction == target).sum().item() / target.numel()

            # Log training accuracy and loss
            train_logger.add_scalar("Accuracy", accuracy, global_step)
            train_logger.add_scalar("Loss", loss.item(), global_step)

            global_step += 1

        # Post epoch accuracy and loss
        epoch_loss = current_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {epoch_loss:.3f}")
        train_logger.add_scalar("epoch_loss", epoch_loss, epoch)

        # Validation
        model.eval()
        confusion_matrix = ConfusionMatrix(size=5)
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, prediction = torch.max(output.data, 1)

                correct += (prediction == target).sum().item()
                total += target.numel()
                print(f"Correct: {correct}, Total: {total}")
                # IOU and confusion Matrix
                confusion_matrix.add(prediction, target)

        # Calculate, log, and print validation accuracy and IOU
        valid_accuracy = (correct / total) * 100
        valid_IOU = confusion_matrix.iou
        valid_logger.add_scalar("Accuracy", valid_accuracy, epoch)
        valid_logger.add_scalar("IOU", valid_IOU, epoch)
        print(
            f"Epoch: {epoch + 1}/{args.epochs}, Validation Accuracy: {valid_accuracy:.2f}%, Validation IOU: {valid_IOU:.3f}")

        if valid_IOU > top_valid_IOU and valid_IOU >= 0.3:
            top_valid_IOU = valid_IOU
            print(f"Saved model with IOU {top_valid_IOU:.3f}")
            save_model(model)

        # Update learning rate
        scheduler.step()


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    if logger is not None:
        logger.add_image('image', imgs[0], global_step)
        logger.add_image('label', dense_transforms.label_to_pil_image(lbls[0].cpu()).convert('RGB'), global_step,
                         dataformats='HWC')
        logger.add_image('prediction',
                         dense_transforms.label_to_pil_image(logits[0].argmax(dim=0).cpu()).convert('RGB'), global_step,
                         dataformats='HWC')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='log')
    parser.add_argument('-m', '--model', choices=['cnn', 'fcn'], default='cnn')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--log_interval', type=int, default=10, help='How often to log training status')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')

    # Put any other custom arguments here

    args = parser.parse_args()
    train(args)
