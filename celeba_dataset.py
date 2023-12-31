import os
import torch
import torch.utils.data as data
import PIL
import numpy as np
import cv2


class CelebaDataset(data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None, target_transform=None, albu=True):
        images = []
        targets = []
        # self.classes = np.array([19, 31, 34, 15, 35])
        self.classes = np.array([19, 31, 34])
        count = 0 
        for line in open(ann_file, 'r'):
            sample = line.split()
            if len(sample) != 41:
                raise (RuntimeError("# Ann   otated face attributes of CelebA dataset should not be different from 40"))
            # print(sample)
            sample = np.array(sample)
            # print(sample[0])
            images.append(sample[0])
            sample_ = sample[1:]
            target_ = [int(i) for i in sample_[self.classes]]
            targets.append(target_)
            if sum(target_) == 3:
                # print(sample[0])
                count+=1
        print("number of class 7", count)

        self.images = [os.path.join(img_dir, img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.albu_transform = albu

    def __getitem__(self, index):
        path = self.images[index]
        if self.albu_transform:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image =  PIL.Image.open(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.transform:
            if self.albu_transform:
                augmented = self.transform(image=image)
                image = augmented['image'] #albu
            else: 
                image = self.transform(image) # torchvision
                
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)
    
class CelebaTestset(data.Dataset):
    def __init__(self, img_dir, transform=None):
        images = []
        imagenames = []
        celeba_ctr = {}
        
        valid_images = [".jpg",".jpeg", ".gif",".png",".tiff"]
        # for dirname in os.listdir(img_dir):
        #     dirpath = os.path.join(img_dir, dirname)
        #     if os.path.isdir(dirpath):
        #         counter = 0
        #         for filename in os.listdir(dirpath):
        #             ext = os.path.splitext(filename)[1]
        #             if ext.lower() not in valid_images:
        #                 continue
        #             images.append(os.path.join(dirpath, filename))
        #             imagenames.append(filename)
        #             counter += 1
        #         celeba_ctr[dirname] = counter
        print("LOADING TEST SET", img_dir)
        for filename in os.listdir(img_dir):
            # print(filename)
            ext = os.path.splitext(filename)[1]
            if ext.lower() not in valid_images:
                continue
            images.append(os.path.join(img_dir, filename))
            imagenames.append(filename)


        self.images = images
        self.imagenames = imagenames
        self.celeba_ctr = celeba_ctr
        self.transform = transform

    def __getitem__(self, index):
        path = self.images[index]
        img_name = self.imagenames[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented['image'] #albu
            except: 
                image = PIL.Image.fromarray(image)
                image = self.transform(image) # torchvision
                
        return image, img_name

    def __len__(self):
        return len(self.images)
    