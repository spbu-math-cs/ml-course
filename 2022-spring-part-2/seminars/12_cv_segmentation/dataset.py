import torch
import cv2
import numpy as np
import pandas as pd
import os


def preprocces_img_and_mask(img, mask):
    img = np.tile(img[..., None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32')
    mx = np.max(img)
    if mx:
        img /= mx

    img = img.transpose(2, 0, 1)

    mask = mask / 255.
    mask = mask.round().astype(np.float32)
    mask = np.transpose(mask, (2, 0, 1))

    return img, mask


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_path = df['mask_path'].tolist()
        self.transforms = transforms

        self.data_path = '/datasets/train_classifier/uw-madison-gi-tract-image-segmentation/png'
        self.mask_path = '/datasets/train_classifier/uw-madison-gi-tract-image-segmentation/np'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(self.data_path + img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (256, 256))

        if self.label:
            msk_path = self.msk_path[index]
            msk = np.load(self.mask_path + msk_path)
            msk = msk.astype('float32')
            msk = cv2.resize(msk, (256, 256))

            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']

            img, msk = preprocces_img_and_mask(img, msk)

            return torch.tensor(img).float(), torch.tensor(msk)

        else:
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)
