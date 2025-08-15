"""
CD Dataset
"""
import os
from PIL import Image
import numpy as np
from torch.utils import data
import data.util as Util
from torch.utils.data import Dataset
import torchvision
import torch

totensor = torchvision.transforms.ToTensor()

"""
CD Dataset 
├─image
├─image_post
├─label
└─list
"""

IMG_FOLDER_NAME = 'A'
IMG_POST_FOLDER_NAME = 'B'
LABEL_FOLDER_NAME = 'label'
LABEL1_FOLDER_NAME = 'label1'
LABEL2_FOLDER_NAME = 'label2'
LIST_FOLDER_NAME = 'list'

label_suffix = ".png"

#list内存放image_name 构建读取图片名字函数
def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

#获取各个文件夹的路径
def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL_FOLDER_NAME, img_name)

def get_label1_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL1_FOLDER_NAME, img_name)

def get_label2_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL2_FOLDER_NAME, img_name)


class CDDataset(Dataset):
    def __init__(self, root_dir, resolution=256, split='train', data_len=-1, label_transform=None):

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_len = data_len
        self.split = split #train / val / test
        self.label_transform = label_transform

        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])

        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')

        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        # Load the third channel of the image as the label
        img_label_rgb = Image.open(L_path).convert('RGB')
        _, _, img_label = img_label_rgb.split()

        img_label = np.array(img_label)
        img_label = (img_label == 255).astype(np.uint8)
        img_label = Image.fromarray(img_label)

        img_A = Util.transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = Util.transform_augment_cd(img_B, min_max=(-1, 1))
        # Convert label to tensor without normalization
        img_label = totensor(img_label) * 255
        img_label = img_label.squeeze().long() # Remove channel dim and convert to long

        return {'A':img_A, 'B':img_B, 'L':img_label, 'Index':index}



class SCDDataset(Dataset):
    def __init__(self, root_dir, resolution=512, split='train', data_len=-1, label_transform=None):

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_len = data_len
        self.split = split #train / val / test
        self.label_transform = label_transform

        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_name = self.img_name_list[index % self.data_len]
        A_path = get_img_path(self.root_dir, img_name)
        B_path = get_img_post_path(self.root_dir, img_name)
        # Handle both Windows and Unix path separators
        name = os.path.basename(A_path).split('.')[0]
        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')

        img_name = self.img_name_list[index % self.data_len]
        L_path = get_label_path(self.root_dir, img_name)
        L1_path = get_label1_path(self.root_dir, img_name)
        L2_path = get_label2_path(self.root_dir, img_name)
        
        try:
            img_label = np.array(Image.open(L_path), dtype=np.uint8)
            img_label1 = np.array(Image.open(L1_path), dtype=np.uint8)
            img_label2 = np.array(Image.open(L2_path), dtype=np.uint8)
        except Exception as e:
            print(f"Error loading label images for {img_name}: {e}")
            # Create empty arrays as fallback
            img_label = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
            img_label1 = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
            img_label2 = np.zeros((self.resolution, self.resolution), dtype=np.uint8)

        # Transform images to tensors with normalization
        img_A = Util.transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = Util.transform_augment_cd(img_B, min_max=(-1, 1))
        
        # Convert labels to tensors
        img_label = torch.from_numpy(img_label)
        img_label1 = torch.from_numpy(img_label1)
        img_label2 = torch.from_numpy(img_label2)
        
        # Handle multi-channel labels by taking first channel if needed
        if img_label.dim() > 2:
            img_label = img_label[0]
            img_label1 = img_label1[0]
            img_label2 = img_label2[0]
        
        # Define number of classes
        num_classes = 7  # This should match your model's num_classes
        max_class_id = num_classes - 1  # For 7 classes (0-6)
        
        # Normalize label values to be within valid range
        img_label = torch.clamp(img_label, 0, max_class_id)
        img_label1 = torch.clamp(img_label1, 0, max_class_id)
        img_label2 = torch.clamp(img_label2, 0, max_class_id)
        
        # Create one-hot encoded class presence vectors
        cls_label1 = torch.zeros(num_classes, dtype=torch.int)
        cls_label2 = torch.zeros(num_classes, dtype=torch.int)
        
        # Get unique class categories
        cls_category1 = torch.unique(img_label1)
        cls_category2 = torch.unique(img_label2)
        
        # Set class presence for label1
        for index in cls_category1:
            idx = int(index)
            if idx < num_classes:
                cls_label1[idx] = 1
            else:
                print(f"Warning: Label index {idx} exceeds number of classes {num_classes}. Skipping.")
        
        # Set class presence for label2
        for index in cls_category2:
            idx = int(index)
            if idx < num_classes:
                cls_label2[idx] = 1
            else:
                print(f"Warning: Label index {idx} exceeds number of classes {num_classes}. Skipping.")

        return {'A':img_A, 'B':img_B, 'L':img_label, 'L1':img_label1, 'L2':img_label2,
                'Index':index, 'name':name, 'cls1':cls_label1, 'cls2':cls_label2}

if __name__ == '__main__':
    # Use a platform-independent path
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_samples')
    print(f"Testing dataset with root directory: {root_dir}")
    
    try:
        # Create dataset instance
        cddata = SCDDataset(root_dir=root_dir)
        print(f"Dataset created with {len(cddata)} samples")
        
        # Test a few samples
        num_samples = min(5, len(cddata))
        for i in range(num_samples):
            try:
                sample = cddata.__getitem__(i)
                print(f"\nSample {i}:")
                print(f"Image A shape: {sample['A'].shape}")
                print(f"Image B shape: {sample['B'].shape}")
                print(f"Label shape: {sample['L'].shape}")
                print(f"Class 1 presence: {sample['cls1']}")
                print(f"Class 2 presence: {sample['cls2']}")
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
    except Exception as e:
        print(f"Error creating or testing dataset: {e}")
