from os import path
from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(args):
    model = CNNClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    train_loader = load_data("data/train", args.num_workers, args.batch_size)
    valid_loader = load_data("data/valid", args.num_workers, args.batch_size)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted_train = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted_train == target).sum().item()

            if batch_idx % args.log_interval == 0:
                print(f"Train Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

        train_accuracy = 100 * correct_train / total_train
        print(f"Training Accuracy after epoch {epoch}: {train_accuracy:.2f}%")

        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                _, predicted_val = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted_val == target).sum().item()

        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Accuracy after epoch {epoch}: {val_accuracy:.2f}%")

        val_loss /= len(valid_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 15:
                print("Early stopping due to no improvement.")
                break

        save_model(model)


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

    args = parser.parse_args()
    train(args)
