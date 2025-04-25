import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np
import nibabel as nib
import glob
join = os.path.join

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])

class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        # data_file_path = os.path.join(root, train_file)
        # with open(data_file_path, 'r') as f:
        #     datalist = [i.strip() for i in f.readlines()]
        # datalist.sort()

        # volpaths = []
        # for dataname in datalist:
        #     volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        '''Yao'''
        patients_dir = glob.glob(join(root, 'vol', '*_vol.npy'))
        patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])
        print('###############', len(patients_dir))
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)
        np.random.seed(0)
        np.random.shuffle(pid_idx)
        n_fold_list = np.split(pid_idx, 3)

        volpaths = []
        for i, fold in enumerate(n_fold_list):
            if i != 0:
                for idx in fold:
                    volpaths.append(patients_dir[idx])
        datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths]
        '''Yao'''

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):
        # Get paths
        volpath = self.volpaths[index]
        name = self.names[index]
        
        # Load volume data
        x = np.load(volpath)
        
        # Load regular segmentation
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        
        # Load GA segmentation
        ga_segpath = volpath.replace('vol', 'GA_seg')
        try:
            ga_y = np.load(ga_segpath)
            # Clip values to valid range [0, num_cls-1]
            ga_y = np.clip(ga_y, 0, self.num_cls - 1)
        except FileNotFoundError:
            logging.warning(f"GA segmentation not found at {ga_segpath}, using zeros")
            ga_y = np.zeros_like(y)
        
        # Add dimensions
        x, y, ga_y = x[None, ...], y[None, ...], ga_y[None, ...]
    
        # Apply transforms to all data
        x, y, ga_y = self.transforms([x, y, ga_y])
    
        # Process volume data
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))  # [Bsize,channels,Height,Width,Depth]
        x = x[:, self.modal_ind, :, :, :]
        
        # Process regular segmentation
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))
        
        # Process GA segmentation similarly
        ga_y = np.reshape(ga_y, (-1))
        ga_one_hot_targets = np.eye(self.num_cls)[ga_y]
        ga_yo = np.reshape(ga_one_hot_targets, (1, H, W, Z, -1))
        ga_yo = np.ascontiguousarray(ga_yo.transpose(0, 4, 1, 2, 3))
    
        # Convert to torch tensors
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)
        ga_yo = torch.squeeze(torch.from_numpy(ga_yo), dim=0).float()
    
        # Get random mask
        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        
        return x, yo, ga_yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, test_file='test.txt'):
        # data_file_path = os.path.join(root, test_file)
        # with open(data_file_path, 'r') as f:
        #     datalist = [i.strip() for i in f.readlines()]
        # datalist.sort()
        # volpaths = []
        # for dataname in datalist:
        #     volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        
        '''Yao'''
        patients_dir = glob.glob(join(root, 'vol', '*_vol.npy'))
        patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)
        np.random.seed(0)
        np.random.shuffle(pid_idx)
        n_fold_list = np.split(pid_idx, 3)

        volpaths = []
        for i, fold in enumerate(n_fold_list):
            if i == 0:
                for idx in fold:
                    volpaths.append(patients_dir[idx])
        datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths]
        '''Yao'''

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])
        self.num_cls = num_cls
    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]
        
        # Load volume data
        x = np.load(volpath)
        
        # Load regular segmentation
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        
        # Load GA segmentation 
        ga_segpath = volpath.replace('vol', 'GA_seg')
        try:
            ga_y = np.load(ga_segpath)
            # Clip values to valid range [0, num_cls-1]
            ga_y = np.clip(ga_y, 0, self.num_cls - 1)
        except FileNotFoundError:
            logging.warning(f"GA segmentation not found at {ga_segpath}, using zeros")
            ga_y = np.zeros_like(y)
        
        # Add dimensions
        x, y, ga_y = x[None, ...], y[None, ...], ga_y[None, ...]
        
        # Apply transforms
        x, y, ga_y = self.transforms([x, y, ga_y])  # Make sure transforms handle 3 inputs
        
        # Process volume data
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        x = x[:, self.modal_ind, :, :, :]
        
        # Process regular segmentation
        y = np.ascontiguousarray(y)
        
        # Process GA segmentation
        ga_shape = ga_y.shape
        if len(ga_shape) == 5:  # (B, C, H, W, D)
            _, _, H, W, Z = ga_shape
        elif len(ga_shape) == 4:  # (B, H, W, D)
            _, H, W, Z = ga_shape
        else:  # (H, W, D)
            H, W, Z = ga_shape
            
        # Ensure GA segmentation is integer type and within valid range
        ga_y = np.reshape(ga_y, (-1)).astype(np.int64)
        ga_y = np.clip(ga_y, 0, self.num_cls - 1)
        
        # Create one-hot encoded targets
        ga_one_hot_targets = np.eye(self.num_cls)[ga_y]
        ga_yo = np.reshape(ga_one_hot_targets, (1, H, W, Z, -1))
        ga_yo = np.ascontiguousarray(ga_yo.transpose(0, 4, 1, 2, 3))
    
        # Convert to tensors
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)
        ga_yo = torch.squeeze(torch.from_numpy(ga_yo), dim=0).float()
    
        return x, y, ga_yo, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, settype='train', modal='all'):
        data_file_path = os.path.join(root, 'val.txt')
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        mask = mask_array[index%15]
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)
