from dataset import SegDataset
from model import UnetOverResnet18
import torch.nn as nn
import torch.utils.data
import datetime
from datetime import timedelta
import sys
import time
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A

train_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf(
            [
                A.GridDistortion(num_steps=5, distort_limit=0.05),
                A.ElasticTransform(alpha=1)
            ], p=0.4
        )
    ]
)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1, 0))

    return dice


def print_one_line(s):
    time_string = datetime.datetime.now().strftime('%H:%M:%S')
    sys.stdout.write('\r' + time_string + ' ' + s)
    sys.stdout.flush()

# print_one_line('Epoch {} Loss {:.6f} | ({}/{})'.format(epoch, train_loss / batch_idx,
#                                                                num_img, len(data_loader) * batch_size))

fold = 0
df = pd.read_csv('train_folds.csv')

train_df = df[df['fold'] != fold]
valid_df = df[df['fold'] == 0]

dataset_train = SegDataset(train_df, transforms=train_aug)
dataset_val = SegDataset(valid_df)

batch = 64

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch, shuffle=True, num_workers=6)

val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch, shuffle=False, num_workers=6)

# model = UnetOverResnet18(pretrained=True)
model = smp.Unet(
    encoder_name='efficientnet-b1',
    encoder_weights='imagenet',
    in_channels=3,
    classes=3,
    activation=None
)
model.cuda()

lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()

for epoch in range(1, 100):
    model.train()

    num_img = 0
    batch_idx = 0
    train_loss = 0

    for images, masks in train_loader:
        optimizer.zero_grad()

        images = images.cuda()
        masks = masks.cuda()

        logit = model(images)

        loss = bce(logit, masks)

        train_loss += loss

        loss.backward()

        optimizer.step()

        num_img += images.size(0)
        batch_idx += 1

        print_one_line('Epoch {} Loss {:.6f} | ({}/{})'.format(epoch, train_loss / batch_idx,
                                                                       num_img, len(train_loader) * batch))

    model.eval()
    preds = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.cuda()
            masks = masks.cuda()
            output = model(images)

            output = nn.Sigmoid()(output)
            val_dice = dice_coef(masks, output).cpu().detach().numpy()
            preds.append(val_dice)

    dice = np.mean(preds, axis=0)
    print('')
    print("Val DICE:", dice)

