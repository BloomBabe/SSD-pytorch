import argparse
from utils.modules.loss import MultiBoxLoss
from ssd import *

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", default= "./JSONdata/", help= "Dataroot directory path")
ap.add_argument("--cfg_root", default= "./utils/architecture_cfg.json", help= "Cfg file path")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 101
cfg_pth = args.cfg_root
checkpoint = args.checkpoint
batch_size = args.batch_size  # batch size
iterations = args.iterations  # number of iterations to train
workers = args.num_workers  # number of workers for loading data in the DataLoader
print_freq = 100  # print training status every __ batches
lr = args.lr  # learning rate
decay_lr_at = [96500, 120000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = args.momentum  # momentum
weight_decay = args.weight_decay
grad_clip = args.grad_clip

global start_epoch, label_map, epoch, checkpoint, decay_lr_at
#Init model or load checkpoint
if checkpoint is None:
    start_epoch= 0
    with open(self.cfg_pth) as f:
        cfgs = json.load(f)
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


