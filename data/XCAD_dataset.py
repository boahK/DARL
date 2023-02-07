import glob
import numpy as np
import os, os.path
import cv2
from torch.utils.data import Dataset
import data.util as Util

class XCADDataset(Dataset):
    def __init__(self, dataroot, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot
        self.data_idx = -1

        if split == 'train':
            # 'trainB'= Image, 'trainC'=Background, 'trainA'=fractal label
            self.A_paths = sorted(glob.glob(os.path.join(dataroot, self.split, 'trainB', '*.png')))
            self.B_paths = sorted(glob.glob(os.path.join(dataroot, self.split, 'trainC', '*.png')))
            self.F_paths = sorted(glob.glob(os.path.join(dataroot, self.split, 'trainA', '*.png')))
            self.data_len = len(self.A_paths)

        elif split == 'val':
            dataPath = os.path.join(dataroot, 'test', 'images')
            dataFiles = sorted(os.listdir(dataPath))[:12]
            for isub, FileName in enumerate(dataFiles):
                self.imageNum.append(FileName)
            self.data_len = len(self.imageNum)
        else:
            dataPath = os.path.join(dataroot, self.split, 'images')
            dataFiles = sorted(os.listdir(dataPath))#[12:]
            for isub, FileName in enumerate(dataFiles):
                self.imageNum.append(FileName)
            self.data_len = len(self.imageNum)

        self.inputSize = 256

    def _random_subsample(self, data):
        opt1 = np.random.randint(0,2)
        opt2 = np.random.randint(0,2)
        data = data[opt1::2, opt2::2]
        return data

    def _shuffle_data_index(self):
        self.data_idx += 1
        if self.data_idx >= self.data_len:
            self.data_idx = 0
            np.random.shuffle(self.A_paths)
            np.random.shuffle(self.B_paths)
            np.random.shuffle(self.L_paths)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.split == 'train':
            self._shuffle_data_index()
            A_path = self.A_paths[index]
            B_path = self.B_paths[index]
            F_path = self.F_paths[index]

            data_A = cv2.imread(A_path, cv2.IMREAD_GRAYSCALE).astype('float')/255.
            data_B = cv2.imread(B_path, cv2.IMREAD_GRAYSCALE).astype('float')/255.
            data_F = cv2.imread(F_path, cv2.IMREAD_GRAYSCALE).astype('float')/255.

            data_A = self._random_subsample(data_A)
            data_B = self._random_subsample(data_B)
            data_F = self._random_subsample(data_F)
        else:
            dataInfo = self.imageNum[index]
            dataPath = os.path.join(self.dataroot, 'test', 'images', dataInfo)
            data_A = cv2.imread(dataPath, cv2.IMREAD_GRAYSCALE).astype('float') / 255.
            labelPath = os.path.join(self.dataroot, 'test', 'masks', dataInfo)
            data_F = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE).astype('float') / 255.
            data_B = data_A
            B_path = dataInfo

        [data_A, data_B, data_F] = Util.transform_augment([data_A, data_B, data_F], split=self.split, min_max=(-1, 1))

        return {'A': data_A, 'B': data_B, 'F': data_F, 'P':B_path, 'Index': index}
