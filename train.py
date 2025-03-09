import argparse
from models.model_DNANet import DNANet
from dataset import *
from log import *
from evaluation_metrics import *
import cv2


parser = argparse.ArgumentParser(description="train LELCM")
parser.add_argument("--task_name", default="train_SIRST_full", type=str, help="task name for saving")
parser.add_argument("--gpu_ids", default=[0], type=list, help="gpus")

parser.add_argument("--train_dataset_name", default="SIRST", type=str, help="dataset name:'SIRST',")
parser.add_argument("--valid_dataset_name", default="SIRST", type=str, help="dataset name:'SIRST',")

parser.add_argument("--label_type", default="full", type=str, help="label type:'centroid', 'coarse', 'full'")
parser.add_argument("--masks_update_path", default="D:/111Project/data/mask_update/")
parser.add_argument("--dataset_dir", default="D:/111Project/data/SIRST/")
parser.add_argument("--batch_size", default=1, help="train batch size")
parser.add_argument("--patch_size", default=256, help="random crop patch size")
parser.add_argument("--num_workers", default=1, help="num of workers for dataloader")
parser.add_argument("--lr", default=0.0005, type=float)
parser.add_argument("--epoch", default=500, help="num of train epoch")
parser.add_argument("--gamma", type=float, default=0.1, help='Gamma lr decay')
parser.add_argument("--steps", type=int, default=[200, 300], help="lr decayed by step, default: [200, 300]")
parser.add_argument("--ckpt_save_frequency", default=1, type=int, help="frequency to save model")
parser.add_argument("--ckpt_save_dir", default="./runs/", type=str, help="checkpoint save directory")

parser.add_argument("--valid_frequency", default=1, type=int, help="frequency of valid")
parser.add_argument("--valid_save", default="./runs/valid/", type=str, help="save_path in valid or None")
parser.add_argument("--valid_threshold", default=0.5, type=float, help="valid threshold")

parser.add_argument("--resume", default=None, type=str, help="resume ckpt path")


global opt
opt = parser.parse_args()


def train():
    train_log = Log(opt.task_name)
    epoch_load = 0
    train_dataset = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.train_dataset_name, label_type=opt.label_type, patch_size=opt.patch_size, masks_update=opt.masks_update_path)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size)
    model = DNANet(mode='train')
    if opt.resume:
        checkpoint = torch.load(opt.resume)
        epoch_load += checkpoint['epoch']
        for step in opt.steps:
            step -= epoch_load
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.steps, gamma=opt.gamma)

    if len(opt.gpu_ids) > 1:
        device_ids = opt.gpu_ids
        model = nn.DataParallel(model, device_ids=device_ids).cuda(device=device_ids[0])
    elif len(opt.gpu_ids) == 1:
        model = model.cuda(0)

    ckpt_save_path = opt.ckpt_save_dir+time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))+"_"+opt.task_name
    os.makedirs(ckpt_save_path, exist_ok=True)

    for epoch_now in range(epoch_load, opt.epoch):
        loss_epoch = []
        for iter_num, (img, mask) in enumerate(train_dataloader):
            model.train()
            img, mask = img.cuda(), mask.cuda()
            output = model(img)
            loss = loss_softIoU(output, mask)
            loss_epoch.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_log.log_print('Epoch---%d, total_loss---%f,' % (epoch_now, float(np.array(loss_epoch).mean())))
        if (epoch_now + 1) % opt.ckpt_save_frequency == 0:
            save_path = ckpt_save_path+"/"+str(epoch_now+1)+".pth.tar"
            state = {'epoch': epoch_now+1,
                     'state_dict': model.state_dict()}
            torch.save(state, save_path)
            train_log.log_print("Epoch---%d, model saved" % epoch_now)

        if (epoch_now+1) % opt.valid_frequency == 0:
            valid(train_log, epoch_now, model)

        scheduler.step()


def valid(log, epoch, model):
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.valid_dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    # model = DNANet(mode='test').cuda()
    # model.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    model.eval()

    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()

    with torch.no_grad():
        for iter_num, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            img = img.cuda()
            pred = model.forward(img)
            pred = pred[3][:, :, :size[0], :size[1]]
            if opt.valid_save:
                img_save_path = opt.valid_save+opt.task_name+"/"+img_dir[0]+"/"
                os.makedirs(img_save_path, exist_ok=True)
                pred_save = (np.squeeze(pred.cpu().numpy())*255).astype(np.uint8)
                cv2.imwrite(img_save_path+str(epoch)+".png", pred_save)
            gt_mask = gt_mask[:, :, :size[0], :size[1]]
            eval_mIoU.update((pred > opt.valid_threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0, 0, :, :] > opt.valid_threshold).cpu(), gt_mask[0, 0, :, :], size)

        results1 = eval_mIoU.get()
        results2 = eval_PD_FA.get()
        # print("pixAcc, mIoU:\t" + str(results1))
        # print("PD, FA:\t" + str(results2))
        log.log_print("pixAcc, mIoU:\t" + str(results1))
        log.log_print("PD, FA:\t" + str(results2))


if __name__ == "__main__":
    train()

