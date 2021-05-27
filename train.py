import datetime as dt
import logging

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from loss import DiceLoss, IoULoss
from unet import UNet
from dataset import Dataset

if __name__ == "__main__":
    now_str = dt.datetime.now().strftime("%m-%d-%Y_%H-%M")

    logging.basicConfig(
        filename=f"./logs/{now_str}_UNet_training.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device : {device}")

    # Loading data
    logging.info("Loading data ...")
    IMAGES = "./dataset/train/images"
    MASKS = "./dataset/train/masks"
    SAMPLES = "./labels/train_labels_GY1QjFw.csv"

    train_samples = pd.read_csv(SAMPLES)

    # Train test split (could be improved in k fold validation)
    train_samples, X_val = train_test_split(
        train_samples, train_size=0.8, random_state=42
    )

    # train_set = SleepApneaDataset(X_TRAIN_PATH, Y_TRAIN_PATH)
    BATCH_SIZE = 128
    CHANNELS = 4
    DROPOUT = 0.1
    N_FEATURES = 9000
    N_EPOCHS = 75
    LEARNING_RATE = 0.001

    logging.info(
        f"Hyperparameters : "
        f"Batch size : {BATCH_SIZE}, "
        f"Epochs : {N_EPOCHS}, "
        f"Learning rate : {LEARNING_RATE}, "
        f"Dropout : {DROPOUT}"
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=Dataset('/labels/train_images_Es8kvkp.csv', TRAIN_IMAGES, TRAIN_MASKS),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=Dataset('/labels/train_images_Es8kvkp.csv', TRAIN_IMAGES, TRAIN_MASKS),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )

    # Def of network
    model = UNet(CHANNELS, DROPOUT)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters for UNet : {n_params}")

    # Optimizer : Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss function
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceLoss()
    # loss_fn = IoULoss()

    # Training :
    for epoch in range(N_EPOCHS):
        model.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x).to(device), Variable(target).to(device)
            out = model(x)
            out = torch.squeeze(out)
            loss = loss_fn.forward(out, target)
            loss.backward()
            optimizer.step()

            # Print performances every 50 batches
            if batch_idx % 64 == 0:
                print(f"Epoch : {epoch}, Batch {batch_idx}, Loss : {loss}")
                logging.info(f"Epoch : {epoch}, Batch {batch_idx}, Loss : {loss}")

        # testing
        model.eval()
        loss_val = 0
        min_loss = 2

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(val_loader):
                x, target = x.to(device), target.to(device)
                out = model(x)
                out = torch.squeeze(out)
                loss = loss_fn(out, target)

                loss_val += loss.item()

                """
                for i, t in enumerate(THRESHOLDS):
                    out = out > t
                    out = out.int()

                    out, target = out.cpu(), target.cpu()
                    dreem_metrics[i] += dreem_sleep_apnea_custom_metric(out, target)
                """

            print(f"Validation loss: {loss_val / len(val_loader)}")
            logging.info(f"Validation loss: {loss_val / len(val_loader)}")

        if min_loss > loss_val:
            best_params = model.state_dict()
            min_loss = loss_val

    now_str = dt.datetime.now().strftime("%m-%d-%Y_%H-%M")
    torch.save(model.state_dict, "./model/" + now_str + "_unet")