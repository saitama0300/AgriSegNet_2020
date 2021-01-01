
import os
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import random
from torchvision import transforms



def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def make_dataset(root, mode):

    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train' or mode=='val':
    
        train_img_path = os.path.join(root, mode, 'images','rgb')
        train_img_path_nir = os.path.join(root, mode, 'images','nir')
        train_mask_path = os.path.join(root, mode, 'labels')

        images = os.listdir(train_img_path)
        labels = os.listdir(os.path.join(train_mask_path,'waterway'))
        
        boundary = os.path.join(root, mode, 'boundaries')
        mask = os.path.join(root,mode,'masks')

        
        for im, lbl in zip(images, labels):
            label_paths = (os.path.join(train_mask_path, 'cloud_shadow',lbl), os.path.join(train_mask_path, 'double_plant',lbl), 
            os.path.join(train_mask_path, 'planter_skip',lbl),os.path.join(train_mask_path, 'standing_water',lbl),os.path.join(train_mask_path, 'waterway',lbl),
            os.path.join(train_mask_path, 'weed_cluster',lbl))
            
            item = (os.path.join(train_img_path, im), os.path.join(train_img_path_nir, im), os.path.join(boundary, lbl), os.path.join(mask, lbl))

            items.append((item,label_paths))
        
    else:
        test_img_path = os.path.join(root, mode, 'images','rgb')
        test_img_path_nir = os.path.join(root, mode, 'images','nir')

        images = os.listdir(test_img_path)
        
        boundary = os.path.join(root, mode, 'boundaries')
        mask = os.path.join(root,mode,'masks')
        file_list = os.listdir(mask)
        
        for im, lbl in zip(images,file_list):
            
            item = (os.path.join(test_img_path, im), os.path.join(test_img_path_nir, im), os.path.join(boundary, lbl), os.path.join(mask, lbl))

            items.append(item)

    return items


class ImageDataset(Dataset):
    
    def __init__(self, mode, root_dir,image_down):

        self.root_dir = root_dir
        self.imgs = make_dataset(root_dir, mode)
        self.mode = mode
    
        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.nir_normalize = transforms.Normalize(
            mean=[0.485],
            std=[0.229]
        )

        self.classes = ['background',
                        'cloud_shadow',
                        'double_plant',
                        'planter_skip',
                        'standing_water',
                        'waterway',
                        'weed_cluster']
        self.img_downsampling_rate  = image_down
        self.segm_downsampling_rate = image_down

        self.img_down_size = lambda img: imresize(
            img,
            (int(img.size[0] / 1.0*self.img_downsampling_rate), int(img.size[1] / 1.0*self.img_downsampling_rate)),
            interp='bilinear')
        
        self.img_down_size2 = lambda img: imresize(
            img,
            (int(img.size[0] /(2*self.img_downsampling_rate)), int(img.size[1] / (2*self.img_downsampling_rate))),
            interp='bilinear')
        
        self.img_down_size3 = lambda img: imresize(
            img,
            (int(img.size[0] / (4*self.img_downsampling_rate)), int(img.size[1] / (4*self.img_downsampling_rate))),
            interp='bilinear')

        self.label_down_size = lambda label: imresize(
            label,
            (int(label.size[0] / self.segm_downsampling_rate), int(label.size[1] / self.segm_downsampling_rate)),
            interp='nearest')
        
        

    def __len__(self):
        return len(self.imgs)

    def img_transform(self, rgb, nir):
        # 0-255 to 0-1
        rgb = np.float32(np.array(rgb)) / 255.
        nir = np.expand_dims(np.float32(np.array(nir)), axis=2) / 255.

        rgb = rgb.transpose((2, 0, 1))
        nir = nir.transpose((2, 0, 1))

        rgb = self.rgb_normalize(torch.from_numpy(rgb))
        nir = self.nir_normalize(torch.from_numpy(nir))
        
        img = torch.cat([rgb, nir], axis=0)

        return img

    def vmask_transform(self, boundary, mask):
        boundary = np.array(boundary) / 255.
        mask = np.array(mask) / 255.

        boundary = torch.from_numpy(boundary).long()
        mask = torch.from_numpy(np.array(mask)).long()

        return boundary * mask

    def label_transform(self, label_imgs):
        labels = [torch.from_numpy(np.array(img) / 255.).long() for img in label_imgs]
        labels = torch.stack(labels, dim=0)

        sumed = labels.sum(dim=0, keepdim=True)
        bg_channel = torch.zeros_like(sumed)
        bg_channel[sumed == 0] = 1

        return torch.cat((bg_channel, labels), dim=0).float()

    def __getitem__(self, index):

        if(self.mode=='train' or self.mode=='val'):
            (rgb_path, nir_path, boundary_path, mask_path), label_paths = self.imgs[index]

            rgb_prime = Image.open(rgb_path).convert('RGB')
            nir_prime = Image.open(nir_path).convert('L')

            assert rgb_prime.size == nir_prime.size
            rgb, nir = self.img_down_size(rgb_prime), self.img_down_size(nir_prime)
            rgb2, nir2 = self.img_down_size2(rgb_prime), self.img_down_size2(nir_prime)
            rgb3, nir3 = self.img_down_size3(rgb_prime), self.img_down_size3(nir_prime)
            
            boundary = Image.open(boundary_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            boundary, mask = self.label_down_size(boundary), self.label_down_size(mask)

            label_imgs = []
            for c in label_paths:
                label_imgs.append(Image.open(c).convert('L'))
            # import ipdb; ipdb.set_trace()

            rgb = np.array(rgb)
            nir = np.array(nir)
            rgb2 = np.array(rgb2)
            nir2 = np.array(nir2)
            rgb3 = np.array(rgb3)
            nir3 = np.array(nir3)
            boundary = np.array(boundary)
            mask = np.array(mask)
            #print(rgb2.shape,nir2.shape)
            label_imgs = [np.array(self.label_down_size(label_img)) for label_img in label_imgs]

            rgb, rgb2, rgb3 = self.randomHueSaturationValue(rgb,rgb2,rgb3)

            rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, label_imgs = self.randomShiftScaleRotate(
                rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, label_imgs)
            #print(rgb2.shape,nir2.shape)
            
            rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, label_imgs = self.randomHorizontalFlip(
                rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, label_imgs)
            
            rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, label_imgs = self.randomVerticalFlip(
                rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, label_imgs)
            
            rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, label_imgs = self.randomRotate90(
                rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, label_imgs)
            
            img = self.img_transform(rgb, nir)
            img2 = self.img_transform(rgb2, nir2)
            img3 = self.img_transform(rgb3, nir3)
            valid_mask = self.vmask_transform(boundary, mask)
            label = self.label_transform(label_imgs)

            info = rgb_path.split("\\")[-1]
            valid_mask= valid_mask.float()
            label *= valid_mask.unsqueeze(dim=0)

        
            l = torch.argmax(label,dim=0)
            return img3.float(), img2.float(), img.float(), l, valid_mask.unsqueeze(dim=0), info
            
        else:

            rgb_path, nir_path, boundary_path, mask_path = self.imgs[index]

            rgb_prime = Image.open(rgb_path).convert('RGB')
            nir_prime = Image.open(nir_path).convert('L')

            assert rgb_prime.size == nir_prime.size
            rgb, nir = self.img_down_size(rgb_prime), self.img_down_size(nir_prime)
            rgb2, nir2 = self.img_down_size2(rgb_prime), self.img_down_size2(nir_prime)
            rgb3, nir3 = self.img_down_size3(rgb_prime), self.img_down_size3(nir_prime)
            
            boundary = Image.open(boundary_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            boundary, mask = self.label_down_size(boundary), self.label_down_size(mask)
            # import ipdb; ipdb.set_trace()

            rgb = np.array(rgb)
            nir = np.array(nir)
            rgb2 = np.array(rgb2)
            nir2 = np.array(nir2)
            rgb3 = np.array(rgb3)
            nir3 = np.array(nir3)
            boundary = np.array(boundary)
            mask = np.array(mask)
            #print(rgb2.shape,nir2.shape)
           
            img = self.img_transform(rgb, nir)
            img2 = self.img_transform(rgb2, nir2)
            img3 = self.img_transform(rgb3, nir3)
            valid_mask = self.vmask_transform(boundary, mask)

            info = rgb_path.split("\\")[-1]
            valid_mask= valid_mask.float()
        
            info = info[:-4]+".png"
            return img3.float(), img2.float(), img.float(), valid_mask, info

    def get_weight(self):
        num_list = [0] * len(self.classes)

        for sample in self.list_sample:
            classes_ = sample['classes']

            for class_ in classes_:
                class_id = self.classes.index(class_)
                num_list[class_id] += 1

        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def reverse_sample(self):
        rand_number, now_sum = np.random.random() * self.sum_weight, 0
        for i in range(len(self.list_sample)):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def _get_class_dict(self):
        class_dict = dict()
        for i, sample in enumerate(self.list_sample):
            classes_ = sample['classes']

            for class_ in classes_:
                class_id = self.classes.index(class_)

                if class_id not in class_dict:
                    class_dict[class_id] = []
                class_dict[class_id].append(i)

        return class_dict

    def randomHueSaturationValue(self, image, image2, image3,
                                 hue_shift_limit=(-180, 180),
                                 sat_shift_limit=(-255, 255),
                                 val_shift_limit=(-255, 255), u=0.5):
        if np.random.random() < u:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
            h2, s2, v2 = cv2.split(image2)
            image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2HSV)
            h3, s3, v3 = cv2.split(image3)

            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            h2 +=hue_shift
            h3 += hue_shift

            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            s2 = cv2.add(s2,sat_shift)
            s3 = cv2.add(s3, sat_shift)

            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
            v = cv2.add(v, val_shift)
            v2 = cv2.add(v2, val_shift)
            v3 = cv2.add(v3, val_shift)

            image = cv2.merge((h, s, v))
            image2 = cv2.merge((h2,s2, v2))
            image3 = cv2.merge((h3, s3, v3))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            image2 = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)
            image3 = cv2.cvtColor(image3, cv2.COLOR_HSV2BGR)

        return image, image2, image3

    def randomShiftScaleRotate(self, rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, labels,
                               shift_limit=(-0.0, 0.0),
                               scale_limit=(-0.0, 0.0),
                               rotate_limit=(-0.0, 0.0),
                               aspect_limit=(-0.0, 0.0),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
        if np.random.random() < u:
            height, width, _ = rgb.shape
            height2, width2, _ = rgb2.shape
            height3, width3, _ = rgb3.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
            dx2 = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width2)
            dy2 = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height2)
            dx3 = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width3)
            dy3 = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height3)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            ######################################################################
            box0_2 = np.array([[0, 0], [width2, 0], [width2, height2], [0, height2], ])
            box1_2 = box0_2 - np.array([width2 / 2, height2 / 2])
            box1_2 = np.dot(box1_2, rotate_matrix.T) + np.array([width2 / 2 + dx2, height2 / 2 + dy2])

            box0_2 = box0_2.astype(np.float32)
            box1_2 = box1_2.astype(np.float32)
            mat2 = cv2.getPerspectiveTransform(box0_2, box1_2)
            ###########################################################################
            box0_3 = np.array([[0, 0], [width3, 0], [width3, height3], [0, height3], ])
            box1_3 = box0_3 - np.array([width3 / 2, height3 / 2])
            box1_3 = np.dot(box1_3, rotate_matrix.T) + np.array([width3 / 2 + dx3, height3 / 2 + dy3])

            box0_3 = box0_3.astype(np.float32)
            box1_3 = box1_3.astype(np.float32)
            mat3 = cv2.getPerspectiveTransform(box0_3, box1_3)
            ###################################################################
            rgb = cv2.warpPerspective(rgb, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                      borderValue=(
                                            0, 0,
                                            0,))
            nir = cv2.warpPerspective(nir, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                      borderValue=(
                                            0))
            rgb2 = cv2.warpPerspective(rgb2, mat2, (width2, height2), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                      borderValue=(
                                            0, 0,
                                            0,))
            nir2 = cv2.warpPerspective(nir2, mat2, (width2, height2), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                      borderValue=(
                                            0))
            rgb3 = cv2.warpPerspective(rgb3, mat3, (width3, height3), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                      borderValue=(
                                            0, 0,
                                            0,))
            nir3 = cv2.warpPerspective(nir3, mat3, (width3, height3), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                      borderValue=(
                                            0))
            ##########################################
            boundary = cv2.warpPerspective(boundary, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                       borderValue=(
                                            0))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                       borderValue=(
                                            0))
            if labels!=None:
                for i in range(len(labels)):
                    labels[i] = cv2.warpPerspective(labels[i], mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,
                                            borderValue=(
                                                    0))

        return rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, labels

    def randomHorizontalFlip(self, rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = cv2.flip(rgb, 1)
            nir = cv2.flip(nir, 1)
            rgb2 = cv2.flip(rgb2, 1)
            nir2 = cv2.flip(nir2, 1)
            rgb3 = cv2.flip(rgb3, 1)
            nir3 = cv2.flip(nir3, 1)
            
            boundary = cv2.flip(boundary, 1)
            mask = cv2.flip(mask, 1)

            if labels!=None:
                for i in range(len(labels)):
                    labels[i] = cv2.flip(labels[i], 1)

        return rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, labels

    def randomVerticalFlip(self, rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = cv2.flip(rgb, 0)
            nir = cv2.flip(nir, 0)
            rgb2 = cv2.flip(rgb2, 0)
            nir2 = cv2.flip(nir2, 0)
            rgb3 = cv2.flip(rgb3, 0)
            nir3 = cv2.flip(nir3, 0)

            boundary = cv2.flip(boundary, 0)
            mask = cv2.flip(mask, 0)

            if labels != None:
                for i in range(len(labels)):
                    labels[i] = cv2.flip(labels[i], 0)

        return rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, labels

    def randomRotate90(self, rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = np.rot90(rgb)
            nir = np.rot90(nir)
            rgb2 = np.rot90(rgb2)
            nir2 = np.rot90(nir2)
            rgb3 = np.rot90(rgb3)
            nir3 = np.rot90(nir3)

            boundary = np.rot90(boundary)
            mask = np.rot90(mask)

            if labels!=None:
                for i in range(len(labels)):
                    labels[i] = np.rot90(labels[i])

        return rgb, nir, rgb2, nir2, rgb3, nir3, boundary, mask, labels
