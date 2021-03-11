import argparse
import torch.optim as optim
from ssd.modules.loss import MultiBoxLoss
from ssd.datasets.hhwd import HHWDataset
from ssd.utils.box_utils import *
from ssd.ssd import *
from ssd.evalute.detect import *
from ssd.evalute.metrics import *
from time import gmtime, strftime


ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", default= "/content/data/", help="Dataroot directory path")
ap.add_argument("--cfg_root", default= "./ssd/configs/architecture_cfg.json", help= "Cfg file path")
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

def clip_grad(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def save_checkpoint(epoch, model, optimizer):
    """
        Save model checkpoint
    """
    if checkpoint is not None:
        pth = os.path.dirname(os.path.abspath(checkpoint))
    else:
        pth = './checkpoints/'
        if not os.path.exists(pth):
            os.makedirs(pth)
    state = {'epoch': epoch, "model": model, "optimizer": optimizer}
    current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    filename = f"model_state_ssd300_{current_time}.pth.tar"
    torch.save(state, os.path.join(pth, filename))

def adjust_lr(optimizer, scale):
    """
        Scale learning rate by a specified factor
        optimizer: optimizer
        scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))  

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

    val_dataset = HHWDataset(data_folder, mode = "test")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                             shuffle=True, collate_fn=combine,
                                             num_workers=workers, pin_memory=True)

    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]
        
    for epoch in range(start_epoch, epochs):
        if epoch in decay_lr_at:
            print("Decay learning rate...")
            adjust_lr(optimizer, decay_lr_to)
        
        # model.train()
        # for i, (images, boxes, labels) in enumerate(train_loader):
        #     image = images.to(device)
        #     boxes = [bbox.to(device) for bbox in boxes]
        #     labels = [label.to(device) for label in labels]

        #     cls_pred, locs_pred = model(images)
        #     loss = criterion(locs_pred, cls_pred, boxes, labels)
        #     # Backward pass
        #     optimizer.zero_grad()
        #     loss.backward()
            
        #     if grad_clip is not None:
        #         clip_grad(optimizer, grad_clip)
                
        #     optimizer.step()

        model.eval()
        metrics_per_batch = list()
        targets = list()
        for i, (images, boxes, labels) in enumerate(val_loader):
            image = images.to(device)
            boxes = [bbox.to(device) for bbox in boxes]
            labels = [label.to(device) for label in labels]
            targets += labels
            with torch.no_grad():
                cls_pred, locs_pred = model(images)
                locs_pred, label_pred, conf_scores = detect(locs_pred, cls_pred, model.default_bboxes)
            metrics_per_batch += compute_statiscs(locs_pred, label_pred, conf_scores, boxes, labels)
        
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics_per_batch))]
        AP, recall, precision = ap_per_class(true_positives, pred_scores, pred_labels, targets)
        print(f"AP: {AP}\nrecall: {recall}\nprecision: {precision}")