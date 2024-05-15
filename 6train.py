import os
import csv
import pdb
import time
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchvision
from tqdm import tqdm, trange


from models.spatial_transforms import *
from models.temporal_transforms import *
from data import dataset_jester, dataset_EgoGesture, dataset_sthv2
import utils as utils
from models import models as TSN_model
import argparse

from PIL import Image


import warnings
warnings.filterwarnings("ignore")


###
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
###

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='2')

    # args for dataloader
    parser.add_argument('--is_train', action="store_true")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--clip_len', type=int, default=8)
    # args for model 
    parser.add_argument('--freeze_pretrained', action='store_true', help='Whether to freeze pretrained model parameters')
    # args for preprocessing
    parser.add_argument('--initial_scale', type=float, default=1,
                        help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int,
                        help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float,
                        help='Scale step for multiscale cropping')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', type=float, default=[5,10,15], nargs="+",
                        help='lr steps for decreasing learning rate') 
    parser.add_argument('--clip_gradient', '--gd', type=int, default=20, help='gradient clip')
    parser.add_argument('--shift_div', default=8, type=int)
    parser.add_argument('--is_shift', action="store_true")
    parser.add_argument('--npb', action="store_true")
    parser.add_argument('--pretrain', type=str, default='imagenet') # 'imagenet' or False
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--dataset', default='EgoGesture', type=str)
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--pretrained', default=None, type=str)
    args = parser.parse_args()
    return args


args = parse_opts()

params = dict()
if args.dataset == 'EgoGesture':
    params['num_classes'] = 4 ##
elif args.dataset == 'jester':
    params['num_classes'] = 4 ##
elif args.dataset == 'sthv2':
    params['num_classes'] = 174  


params['epoch_num'] = args.epochs
params['batch_size'] = args.batch_size
params['num_workers'] = args.num_workers
params['learning_rate'] = args.lr
params['momentum'] = 0.9
params['weight_decay'] = args.weight_decay
# params['display'] = 20
params['display'] = 1
params['pretrained'] = None
params['recover_from_checkpoint'] = None
params['log'] = 'log'
params['save_path'] = '{}-{}'.format(args.dataset, args.base_model)
params['clip_len'] = args.clip_len
params['frame_sample_rate'] = 1




annot_path = './data/{}_annotation'.format(args.dataset)
# label_path = '/home/raid/zhengwei/{}/'.format(args.dataset) # for submitting testing results

# os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = 'cuda:0'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    # print("start training_func...")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    
    model.train()
    # print("model.train() done")
    end = time.time()
    # print(f"end:{end}")
    
    for step, inputs in enumerate(train_dataloader):
        print(f"step:{step}")
        data_time.update(time.time() - end)
        if args.dataset == 'EgoGesture' or args.dataset == 'nvGesture':
            rgb, depth, labels = inputs[0], inputs[1], inputs[2]
            rgb = rgb.to(device, non_blocking=True).float()
            depth = depth.to(device, non_blocking=True).float()
            outputs = model(rgb)
        else:
            rgb, labels = inputs[0], inputs[1]
            rgb = rgb.to(device, non_blocking=True).float()
            print(f"rgb device:{rgb.device}")
            outputs = model(rgb)
        labels = labels.to(device, non_blocking=True).long()
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        
        prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
        losses.update(loss.item(), labels.size(0))
        top1.update(prec1.item(), labels.size(0))
        top3.update(prec3.item(), labels.size(0))

        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        
        
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (step+1) % params['display'] == 0:
            print_string = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}, '
                             'data_time: {data_time.val:.3f} ({data_time.avg:.3f}), batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                             'loss: {loss.val:.4f} ({loss.avg:.4f}), '
                             'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                             'Top-3: {top3_acc.val:.2f} ({top3_acc.avg:.2f})'
                             .format(epoch, step+1, len(train_dataloader),
                                      lr = optimizer.param_groups[2]['lr'],
                                      data_time = data_time, batch_time=batch_time,
                                      loss = losses, top1_acc = top1, top3_acc = top3
                                      )
                            )
            print(print_string)
    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top3_acc_epoch', top3.avg, epoch)
    
def createConfusionMatrix(y_pred, y_true):
    
    # constant for classes
    classes = ('emotion_1','emotion_2','eomtion_3','emotion_4')
    
    # If y_true is a list of tensors, convert each tensor to CPU
    if isinstance(y_true[0], torch.Tensor):
        y_true_cpu = [tensor.cpu() for tensor in y_true]
    else:
        y_true_cpu = y_true

    # If y_pred is a tensor, move it to CPU
    if isinstance(y_pred, torch.Tensor):
        y_pred_cpu = y_pred.cpu()
    else:
        y_pred_cpu = y_pred
        
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true_cpu, y_pred_cpu)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()


def validation(model, val_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(val_dataloader):
            data_time.update(time.time() - end)

            if args.dataset == 'EgoGesture' or args.dataset == 'nvGesture':
                rgb, depth, labels = inputs[0], inputs[1], inputs[2]
                rgb = rgb.to(device, non_blocking=True).float()
                depth = depth.to(device, non_blocking=True).float()
                outputs = model(rgb)
            else:
                rgb, labels = inputs[0], inputs[1]
                rgb = rgb.to(device, non_blocking=True).float()
                outputs = model(rgb)
            labels = labels.to(device, non_blocking=True).long()
            print(f"outputs shape: {outputs.shape}, labels shape: {labels.shape}")
            print(f"outputs device: {outputs.device}, labels device: {labels.device}")
            print(f"min label: {labels.min()}, max label: {labels.max()}")
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
            losses.update(loss.item(), labels.size(0))
            top1.update(prec1.item(), labels.size(0))
            top3.update(prec3.item(), labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (step + 1) % params['display'] == 0:
                print_string = ('Test: [{0}][{1}], '
                                'data_time: {data_time.val:.3f} ({data_time.avg:.3f}), batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                                'loss: {loss.val:.4f} ({loss.avg:.4f}), '
                                'Top-1: {top1_acc.val:.2f} ({top1_acc.avg:.2f}), '
                                'Top-3: {top3_acc.val:.2f} ({top3_acc.avg:.2f})'
                                .format(step+1, len(val_dataloader),
                                        data_time = data_time, batch_time=batch_time,
                                        loss = losses, top1_acc = top1, top3_acc = top3
                                        )
                                )
                print(print_string)
        print_string = ('Testing Results: loss {loss.avg:.5f}, Top-1 {top1.avg:.3f}, Top-3 {top3.avg:.3f}'
                        .format(loss=losses, top1=top1, top3=top3)
                        )
        print(print_string)


    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top3_acc_epoch', top3.avg, epoch)

    model.train()
    return losses.avg, top1.avg

def testing(model, val_dataloader, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(val_dataloader)):
            if args.dataset == 'EgoGesture' or args.dataset == 'nvGesture':
                rgb, depth, labels = inputs[0], inputs[1], inputs[2]
                rgb = rgb.to(device, non_blocking=True).float()
                depth = depth.to(device, non_blocking=True).float()
                outputs = model(rgb)
            else:
                rgb, labels = inputs[0], inputs[1]
                rgb = rgb.to(device, non_blocking=True).float()
                outputs = model(rgb)

            labels = labels.to(device, non_blocking=True).long()
            loss = criterion(outputs, labels)
            prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
            losses.update(loss.item(), labels.size(0))
            top1.update(prec1.item(), labels.size(0))
            top3.update(prec3.item(), labels.size(0))
            batch_time.update(time.time() - end)
            
    print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
    print(print_string)
    
    print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-3 accuracy: {top3_acc:.2f}%'.format(
        top1_acc=top1.avg,
        top3_acc=top3.avg)
    print(print_string)
    



def main():
    best_acc = 0.
    seed = 1
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


    input_mean=[.485, .456, .406]
    input_std=[.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)
        
    scales = [1, .875, .75, .66]
    if args.dataset == 'sthv2':
        trans_train  = torchvision.transforms.Compose([
                                GroupMultiScaleCrop(224, scales),
                                Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                                normalize
                                ])
        temporal_transform_train = torchvision.transforms.Compose([
                                            TemporalUniformCrop_train(args.clip_len)
                                            ])    
        trans_test  = torchvision.transforms.Compose([
                                GroupScale(256),
                                GroupCenterCrop(224),
                                Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                                normalize
                                ])
        temporal_transform_test = torchvision.transforms.Compose([
                                            TemporalUniformCrop_val(args.clip_len)
                                        ])
    elif args.dataset == 'jester':
        trans_train  = torchvision.transforms.Compose([
                                GroupScale(256),
                                GroupMultiScaleCrop(224, scales),
                                Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                                normalize
                                ])
        temporal_transform_train = torchvision.transforms.Compose([
                                            TemporalUniformCrop_train(args.clip_len)
                                            ])    
        trans_test  = torchvision.transforms.Compose([
                                GroupScale(256),
                                GroupCenterCrop(224),
                                Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                                normalize
                                ])
        temporal_transform_test = torchvision.transforms.Compose([
                                            TemporalUniformCrop_val(args.clip_len)
                                        ])
    elif args.dataset == 'EgoGesture':  
        trans_train  = torchvision.transforms.Compose([
                                GroupScale([224, 224]),
                                GroupMultiScaleCrop([224, 224], scales),
                                Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                                ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                                normalize])

        temporal_transform_train = torchvision.transforms.Compose([
                TemporalUniformCrop_train(args.clip_len)
                ])    

        trans_test  = torchvision.transforms.Compose([
                            GroupScale([224, 224]),
                            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                            normalize])

        temporal_transform_test = torchvision.transforms.Compose([
                TemporalUniformCrop_val(args.clip_len)
                ])
    else:
      print("dataset not found")

    criterion = nn.CrossEntropyLoss().to(device)
    if args.is_train: 
        cudnn.benchmark = True
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        print("Loading dataset")
        if args.dataset == 'EgoGesture': 
            train_dataset = dataset_EgoGesture.dataset_video(annot_path, 'train', 
                                                            spatial_transform=trans_train, temporal_transform = temporal_transform_train)
            train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])
            val_dataset = dataset_EgoGesture.dataset_video(annot_path, 'val', spatial_transform=trans_test, temporal_transform = temporal_transform_test)
            val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=params['num_workers'])
        elif args.dataset == 'jester':
            train_dataset = dataset_jester.dataset_video(annot_path, 'train', 
                                                        spatial_transform=trans_train, temporal_transform = temporal_transform_train)
            train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])
            val_dataloader = DataLoader(dataset_jester.dataset_video(annot_path, 'val', 
                                                                    spatial_transform=trans_test, temporal_transform = temporal_transform_test), 
                                        batch_size=params['batch_size'], num_workers=params['num_workers'])
        elif args.dataset == 'sthv2': 
            train_dataset = dataset_sthv2.dataset_video(annot_path, 'train', spatial_transform=trans_train, temporal_transform = temporal_transform_train)
            train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])
            val_dataset = dataset_sthv2.dataset_video(annot_path, 'val', spatial_transform=trans_test, temporal_transform = temporal_transform_test)
            val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=params['num_workers'])    


        print("load model")
        model = TSN_model.TSN(params['num_classes'], args.clip_len, 'RGB', 
                              is_shift = args.is_shift,
                              partial_bn = args.npb,
                              base_model=args.base_model, 
                              shift_div = args.shift_div, 
                              dropout=args.dropout, 
                              img_feature_dim = 224,
                              pretrain=args.pretrain, # 'imagenet' or False
                              consensus_type='avg',
                              fc_lr5 = True)

        print(f"model:{model}")
        if params['pretrained'] is not None:
            checkpoint = torch.load(params['pretrained'], map_location='cpu')
            try:
                model_dict = model.module.state_dict()
            except AttributeError:
                model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and 'fc' not in k}
            print("load pretrained model {}".format(params['pretrained']))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if params['recover_from_checkpoint'] is not None:
            checkpoint = torch.load(params['recover_from_checkpoint'], map_location='cpu')
            try:
                model_dict = model.module.state_dict()
            except AttributeError:
                model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            print("recover from checkpoint {}".format(params['recover_from_checkpoint']))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        
        # if fine-tuning, freeze the pretrained model
        if args.freeze_pretrained:
            #freeze
            for param in model.parameters():
                param.requires_grad = False

            # unnfreeze 
            # for param in model.Sequential.parameters():
            #     param.requires_grad = True
            for param in model.new_fc.parameters():
                param.requires_grad = True
                
            for param in model.consensus.parameters():
                param.requires_grad = True

        # # 定义优化器，只优化解冻的参数
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        
        print(f"device name:{torch.cuda.get_device_name(device)}")
        print(f"device(cuda):{device}")
        policies = model.get_optim_policies()
        # model = nn.DataParallel(model)  # multi-Gpu
        model = model.to(device)
        # print(f"model:{model}")
        if params['recover_from_checkpoint'] is not None:
            testing(model, val_dataloader, criterion)

        for param_group in policies:
            param_group['lr'] = args.lr * param_group['lr_mult']
            param_group['weight_decay'] = args.weight_decay * param_group['decay_mult']
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        # print("after load model...")
        print(model.check.device)
        optimizer = optim.SGD(policies, momentum=params['momentum'])

        logdir = os.path.join(params['log'],params['save_path']+f"_cliplen_{str(params['clip_len'])}_lr_{str(params['learning_rate'])}_"+cur_time)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = SummaryWriter(log_dir=logdir)
        #######
        save_name= f"_cliplen_{str(params['clip_len'])}_lr_{str(params['learning_rate'])}_"+cur_time
        model_save_dir = os.path.join("saved_model",params['save_path']+save_name)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        print("== Start Training... ")
        for epoch in trange(params['epoch_num']):#trange
            train(model, train_dataloader, epoch, criterion, optimizer, writer)
            
            if epoch % 1 == 0:
                val_loss, val_acc = validation(model, val_dataloader, epoch, criterion, optimizer, writer)
                if val_acc > best_acc:
                    checkpoint = os.path.join(model_save_dir,
                                              "accuracy_"+str(val_acc) +
                                             save_name + "frame_sample_rate_" +
                                            str(params['frame_sample_rate'])+ "_checkpoint" + ".pth.tar")
                    utils.save_checkpoint(model, optimizer, checkpoint)
                    best_acc = val_acc
                print('Best Top-1: {:.2f}'.format(best_acc))
               
            
            utils.adjust_learning_rate(params['learning_rate'], optimizer, epoch, args.lr_steps)
        writer.close
    


if __name__ == '__main__':
    print(f"== cuda.is_available:{torch.cuda.is_available()}")
    print("== Entering main...")
    main()
