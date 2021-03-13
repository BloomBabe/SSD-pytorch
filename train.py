import argparse
import os
import time
import datetime

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ssd.modules.loss import MultiBoxLoss
from ssd.datasets.augmentation import SSDDetectAug
from ssd.datasets.hhwd import HHWDataset
from ssd.utils.box_utils import *
from ssd.ssd import *
from ssd.evalute.detect import *
from ssd.evalute.metrics import compute_statiscs, compute_mAP
from time import gmtime, strftime


ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", default= "/content/data/", help="Dataroot directory path")
ap.add_argument("--cfg_root", default= "./ssd/configs/architecture_cfg.json", help= "Cfg file path")
ap.add_argument("--expdir_root", default= "./experiments/", help= "Experiments dir root")
ap.add_argument("--batch_size", default= 8, type= int, help= "Batch size for training")
ap.add_argument("--num_workers", default= 1, type= int, help = "Number of workers")
ap.add_argument("--lr", "--learning-rate", default= 1e-3, type= float, help= "Learning rate")
ap.add_argument("--momentum", default= 0.9, type= float, help = "Momentum value for optim")
ap.add_argument("--weight_decay", default= 5e-4, type= float, help = "Weight decay for SGD")
ap.add_argument("--checkpoint", default = None, help = "path to model checkpoint")
ap.add_argument("--iterations", default= 145000, type= int, help = "number of iterations to train")
ap.add_argument("--grad_clip", default = None, help= "Gradient clip for large batch_size")
ap.add_argument("--adjust_optim", default = None, help = "Adjust optimizer for checkpoint model")
args = ap.parse_args()

global start_epoch, label_map, epoch, checkpoint, decay_lr_at

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = args.dataset_root
expdir_root = args.expdir_root
num_classes = 5
cfg_pth = args.cfg_root
checkpoint = args.checkpoint
batch_size = args.batch_size  # Batch size
iterations = args.iterations  # Number of iterations to train
workers = args.num_workers    # Number of workers for loading data in the DataLoader
print_freq = 100              # Print training status every __ batches
lr = args.lr                  # Learning rate
decay_lr_at = [96500, 120000] # Decay learning rate after these many iterations
decay_lr_to = 0.1             # Decay learning rate to this fraction of the existing learning rate
momentum = args.momentum      # Momentum
weight_decay = args.weight_decay
grad_clip = args.grad_clip

def create_exp_dir(path=expdir_root):
    checkpoint_dir = os.path.join(path, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tensorboard_dir = os.path.join(path, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    return checkpoint_dir, tensorboard_dir

def clip_grad(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def save_checkpoint(epoch, model, optimizer, path = './experiments/checkpoints/'):
    """
        Save model checkpoint
    """
    state = {'epoch': epoch, "model": model, "optimizer": optimizer}
    current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    filename = f"ssd300_epoch{epoch}_lr{lr}_{current_time}.pth.tar"
    torch.save(state, os.path.join(path, filename))

def adjust_lr(optimizer, scale):
    """
        Scale learning rate by a specified factor
        optimizer: optimizer
        scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))  

class Metrics(object):
    def __init__(self):
        super(Metrics, self).__init__()
        self.reset()
    
    def reset(self):
        self.mean_loss = 0.
        self.mean_conf_loss = 0.
        self.mean_loc_loss = 0.
        self.metrics_per_batch = list()
        self.targets = list()

    def update(self, loss, loc_loss, conf_loss, metrics):
        self.mean_loss += loss
        self.mean_conf_loss += conf_loss
        self.mean_loc_loss += loc_loss
        self.metrics_per_batch += metrics
    
    def mean_metrics(self, length):
        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*self.metrics_per_batch))]
        self.targets = torch.LongTensor(self.targets).to(device)
        return (self.mean_loss/length, 
                self.mean_conf_loss/length, 
                self.mean_loc_loss/length, 
                true_positives, 
                pred_scores, 
                pred_labels)

if __name__ == '__main__':
    # Init model or load checkpoint
    if checkpoint is None:
        start_epoch= 0
        with open(cfg_pth) as f:
            cfg = json.load(f)
        model = SSD(num_classes=num_classes, cfgs=cfg, device=device)
        optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum, 
                            weight_decay = weight_decay)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1   
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        if args.adjust_optim is not None:
            print("Adjust optimizer....")
            print(args.lr)
            optimizer = optim.SGD(model.parameters(),lr = lr, momentum = momentum, 
                                weight_decay = weight_decay)
    model = model.to(device)
    criterion = MultiBoxLoss(model.default_bboxes, device=device).to(device)

    train_dataset = HHWDataset(data_folder, mode = "train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               shuffle=True, collate_fn=combine,
                                               num_workers=workers, pin_memory=True)

    val_dataset = HHWDataset(data_folder, mode = "test", transform=SSDDetectAug())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                             shuffle=True, collate_fn=combine,
                                             num_workers=workers, pin_memory=True)

    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]
    
    ckpts_dir, log_dir = create_exp_dir()
    writer = SummaryWriter(log_dir=log_dir, max_queue=20)

    for epoch in range(start_epoch, epochs):
        arrow = int(epoch//(epochs/50))
        print(f'Epoch: {epoch}/{epochs} [{"".join(["=" for i in range(arrow)])}>{"".join(["-" for i in range(50-arrow)])}]')
        if epoch in decay_lr_at:
            print("Decay learning rate...")
            adjust_lr(optimizer, decay_lr_to)
        # =================Training=================
        start_time = time.time()
        metrics = Metrics()
        model.train()
        for i, (images, boxes, labels) in enumerate(train_loader):
            print(f"  [{i}/{len(train_loader)}]")
            for label in labels:
                metrics.targets += label.tolist() 
            image = images.to(device)
            boxes = [bbox.to(device) for bbox in boxes]
            labels = [label.to(device) for label in labels]
            # Loss
            cls_pred, locs_pred = model(images)
            loss, loc_loss, conf_loss = criterion(locs_pred, cls_pred, boxes, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                clip_grad(optimizer, grad_clip)
            optimizer.step()
            # Accuracy
            with torch.no_grad():
                locs_pred, label_pred, conf_scores = detect(locs_pred, cls_pred, model.default_bboxes, image_size=(image.size(2), image.size(3)))
                stats = compute_statiscs(locs_pred, label_pred, conf_scores, boxes, labels)
                metrics.update(loss, loc_loss, conf_loss, stats)
            print ("\033[A                             \033[A")
        train_time = time.time()-start_time 
        print("Time: ", datetime.timedelta(seconds=train_time))

        mean_loss, mean_conf_loss, mean_loc_loss, true_positives, pred_scores, pred_labels = metrics.mean_metrics(len(train_loader))
        metric_dict, mAP = compute_mAP(true_positives, pred_scores, pred_labels, metrics.targets, val_dataset.cat_dict)
        metrics.reset()

        writer.add_scalars("Loss/train_loss", mean_loss, epoch)
        writer.add_scalars("Loss/train_conf_loss", mean_conf_loss, epoch)
        writer.add_scalars("Loss/train_loc_loss", mean_loc_loss, epoch)
        writer.add_scalars("Accuracy/train_per_class", metric_dict, epoch)
        writer.add_scalars("Accuracy/train_mAP50", mAP, epoch)

        print("Loss train_loss: ", mean_loss," Loss train_conf_loss: ", mean_conf_loss, " Loss train_loc_loss: ", mean_loc_loss, " Accuracy train_mAP50: ", mAP)
        save_checkpoint()
        # =================Validation=================
        # model.eval()
        # metrics_per_batch = list()
        # targets = list()
        # for i, (images, boxes, labels) in enumerate(val_loader):
        #     for label in labels:
        #         targets += label.tolist() 
        #     image = images.to(device)
        #     boxes = [bbox.to(device) for bbox in boxes]
        #     labels = [label.to(device) for label in labels]
            
        #     with torch.no_grad():
        #         cls_pred, locs_pred = model(images)
        #         locs_pred, label_pred, conf_scores = detect(locs_pred, cls_pred, model.default_bboxes, image_size=(image.size(2), image.size(3)))
        #     metrics_per_batch += compute_statiscs(locs_pred, label_pred, conf_scores, boxes, labels)
        
        # true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*metrics_per_batch))]
        # targets = torch.LongTensor(targets).to(device)
        # metric_dict = compute_mAP(true_positives, pred_scores, pred_labels, targets, val_dataset.cat_dict)
        # print(metric_dict)