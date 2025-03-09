import os
from utils import *
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, label_type, patch_size, masks_update):
        super(TrainSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.transform = augumentation()
        self.masks_update = masks_update
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()

        self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        self.dataset_name = dataset_name
        # 复制并保存label并更新
        self.label_type = label_type

        if os.path.exists(masks_update):
            shutil.rmtree(masks_update)
        os.makedirs(masks_update)
        for img_idx in self.train_list:
            shutil.copyfile(self.dataset_dir + '/' + '/masks_' + self.label_type + '/' + img_idx,
                            masks_update + '/' + img_idx)

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx]).convert('I')
        mask = Image.open(self.masks_update + '/' + self.train_list[idx])
        mask = np.array(mask, dtype=np.float32) / 255.0

        img = Normalize(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.transform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

        self.img_norm_cfg = get_img_norm_cfg(test_dataset_name, dataset_dir)

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.test_list[idx] ).convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] )

        img = Normalize(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img, 32)
        mask = PadImg(mask, 32)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class TestSetLoader_withimg(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

        self.img_norm_cfg = get_img_norm_cfg(test_dataset_name, dataset_dir)

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.test_list[idx] ).convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] )

        mask = np.array(mask, dtype=np.float32) / 255.0
        img = np.array(img, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        # img = PadImg(img, 32)
        # mask = PadImg(mask, 32)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32), [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class Update_mask(Dataset):
    def __init__(self, dataset_dir, dataset_name, label_type, masks_update):
        super(Update_mask).__init__()
        self.label_type = label_type
        self.masks_update = masks_update
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()

        self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)

    def __getitem__(self, idx):
        img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx]).convert('I')
        mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx])

        mask_update = Image.open(self.masks_update + '/' + self.train_list[idx])
        update_dir = self.masks_update + '/' + self.train_list[idx]
        mask_update = np.array(mask_update, dtype=np.float32) / 255.0
        if len(mask_update.shape) > 2:
            mask_update = mask_update[:, :, 0]

        img = Normalize(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape
        img = PadImg(img, 32)
        mask = PadImg(mask, 32)
        mask_update = PadImg(mask_update, 32)

        img, mask, mask_update = img[np.newaxis, :], mask[np.newaxis, :], mask_update[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        mask_update = torch.from_numpy(np.ascontiguousarray(mask_update))
        return img, mask, mask_update, update_dir, [h, w], self.train_list[idx]

    def __len__(self):
        return len(self.train_list)


def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img


class augumentation(object):
    # data augumentation
    def __call__(self, image, target):
        if random.random()<0.5:
            image = image[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            image = image[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            image = image.transpose(1, 0)
            target = target.transpose(1, 0)
        return image, target
