import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.model_DNANet import DNANet
from dataset import *
import matplotlib.pyplot as plt
import os
from evaluation_metrics import *
import time
from log import *
import numpy as np
import cv2
from models.TLLCM import TLLCM
from models.Top_Hat import Top_Hat

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch LELCM test")
parser.add_argument("--task_name", default="top_hat_test", type=str, help="task of name for saving")
parser.add_argument("--pth_dir", default="./runs/2023-10-08 16_03_38_train_SIRST_full/2.pth.tar", type=str, help="path to pth")
parser.add_argument("--train_dataset_name", default="SIRST", type=str, help="train dataset name")
parser.add_argument("--test_dataset_name", default="SIRST", type=str, help="test dataset name")


parser.add_argument("--dataset_dir", default="D:/111Project/data/SIRST/", type=str, help="data directory")
parser.add_argument("--batch_size", default=1, type=int, help="test batch size")
parser.add_argument("--num_workers", default=1, type=int, help="num of workers for dataloader")
parser.add_argument("--save", default=True, type=bool, help="save images or not")
parser.add_argument("--save_path", default="./runs/pred/", type=str, help="image save path")
parser.add_argument("--threshold", default=0.5, type=float, help="detection threshold")

global opt
opt = parser.parse_args()


def test(log):
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    model = DNANet(mode='test').cuda()
    model.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    model.eval()

    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()

    with torch.no_grad():
        for iter_num, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            img = img.cuda()
            pred = model.forward(img)
            pred = pred[:, :, :size[0], :size[1]]
            if opt.save:
                img_save_path = opt.save_path+opt.task_name+"/"+img_dir[0]
                os.makedirs(opt.save_path+opt.task_name+"/", exist_ok=True)
                pred_save = (np.squeeze(pred.cpu().numpy())*255).astype(np.uint8)
                cv2.imwrite(img_save_path, pred_save)
            gt_mask = gt_mask[:, :, :size[0], :size[1]]
            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)

        results1 = eval_mIoU.get()
        results2 = eval_PD_FA.get()
        # print("pixAcc, mIoU:\t" + str(results1))
        # print("PD, FA:\t" + str(results2))
        log.log_print("pixAcc, mIoU:\t" + str(results1) + '\n')
        log.log_print("PD, FA:\t" + str(results2) + '\n')


def test_TLLCM(log):
    test_set = TestSetLoader_withimg(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)


    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()

    with torch.no_grad():
        for iter_num, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            pred_ = TLLCM(img, cell_sizes=[5, 7, 9, 11, 13])
            pred = torch.tensor(pred_/255.0).unsqueeze(0).unsqueeze(0)
            # gt_mask = torch.tensor(gt_mask_).unsqueeze(0).unsqueeze(0)
            if opt.save:
                img_save_path = opt.save_path+opt.task_name+"/"+img_dir[0]
                os.makedirs(opt.save_path+opt.task_name+"/", exist_ok=True)
                pred_save = (np.squeeze(pred.cpu().numpy())*255).astype(np.uint8)
                cv2.imwrite(img_save_path, pred_save)
            gt_mask = gt_mask[:, :, :size[0], :size[1]]
            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
            print("1")

            results1 = eval_mIoU.get()
            results2 = eval_PD_FA.get()
        # print("pixAcc, mIoU:\t" + str(results1))
        # print("PD, FA:\t" + str(results2))
            log.log_print("pixAcc, mIoU:\t" + str(results1) + '\n')
            log.log_print("PD, FA:\t" + str(results2) + '\n')


def test_Top_Hat(log):
    test_set = TestSetLoader_withimg(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)


    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()

    with torch.no_grad():
        for iter_num, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            pred_ = Top_Hat((np.squeeze(img.cpu().numpy())*255).astype(np.uint8))
            pred = torch.tensor(pred_*1.0).unsqueeze(0).unsqueeze(0)
            # gt_mask = torch.tensor(gt_mask_).unsqueeze(0).unsqueeze(0)
            if opt.save:
                img_save_path = opt.save_path+opt.task_name+"/"+img_dir[0]
                os.makedirs(opt.save_path+opt.task_name+"/", exist_ok=True)
                pred_save = (np.squeeze(pred.cpu().numpy())*255).astype(np.uint8)
                cv2.imwrite(img_save_path, pred_save)
            gt_mask = gt_mask[:, :, :size[0], :size[1]]
            eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
            print("1")

            results1 = eval_mIoU.get()
            results2 = eval_PD_FA.get()
        # print("pixAcc, mIoU:\t" + str(results1))
        # print("PD, FA:\t" + str(results2))
            log.log_print("pixAcc, mIoU:\t" + str(results1) + '\n')
            log.log_print("PD, FA:\t" + str(results2) + '\n')


if __name__ == "__main__":
    test_log = Log(opt.task_name)
    test_Top_Hat(test_log)
