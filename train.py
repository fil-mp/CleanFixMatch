import argparse
import logging
import math
import os
import random
import shutil
import time
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from dataset.cifar import *
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0

import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

class o2u_(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(3*32*32, 1024)
        self.hidden2 = nn.Linear(1024, 512)
        #self.act=nn.Sigmoid()
        self.output = nn.Linear(512, 10)
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(-1, 3*32*32)
        #print(x.shape)
        x=self.hidden1(x)
        x=F.relu(x)
        x=self.hidden2(x)
        x=F.relu(x)
        #x=self.act(x)
        x=self.output(x)
        #x=self.softmax(x)
        return x

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    np.random.seed(10)
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)       

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet','cnn', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--cl-batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--threshold1', default=0.97, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=1024, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')


    args = parser.parse_args()
    global best_acc



    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        
        elif args.arch == 'cnn':
            import models.cnn as models
            model = models.cifar_cnn(num_classes=args.num_classes,
                                      isL2 = False, double_output = False)
        
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    labeled_dataset, unlabeled_dataset, test_dataset, unq_labeled_dataset, unq_unlabeled_dataset, unq_labeled_idx, unq_unlabeled_idx, labels = DATASET_GETTERS[args.dataset](
        args, './data')

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    unq_labeled_trainloader = DataLoader(
        unq_labeled_dataset,
        sampler=SequentialSampler(unq_labeled_dataset),
        batch_size=20,
        num_workers=args.num_workers,
        drop_last=True)

    unq_unlabeled_trainloader = DataLoader(
        unq_unlabeled_dataset,
        sampler=train_sampler(unq_unlabeled_dataset),
        batch_size=20,
        num_workers=args.num_workers,
        drop_last=True)

    #print(labeled_dataset)
    #for batch_idx,(inputs,targets, index) in enumerate (unq_labeled_trainloader):
     # print(index.item())

    #concate_dataset = torch.utils.data.ConcatDataset([unq_labeled_dataset, unq_unlabeled_dataset])
    '''
    train_loader = DataLoader(
             concate_dataset,
             sampler=SequentialSampler(concate_dataset),
             batch_size=125,
             num_workers=args.num_workers)
    '''
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    #initial_idxs=unq_labeled_idx
    all_targets=labels[unq_labeled_idx]
    print(labels.shape)
    model.zero_grad()
    train(args, all_targets, labeled_trainloader, unlabeled_trainloader,unq_unlabeled_trainloader,unq_labeled_dataset, test_loader,unq_labeled_idx,labels,
          model, optimizer, ema_model,scheduler, writer)

def train(args,all_targets,labeled_trainloader, unlabeled_trainloader, unq_unlabeled_trainloader,unq_labeled_dataset, test_loader,unq_labeled_idx,labels,
          model, optimizer, ema_model,scheduler, writer):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_probs = AverageMeter()
    mask1_probs = AverageMeter()
    #l_r = AverageMeter()
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    temp=0.0
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        inds=[]
        lbs=[]
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x, _ = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, _ = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _, i = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _, i = unlabeled_iter.next()
            #print(targets_x)
            
            #adjust_learning_rate(optimizer, epoch, batch_idx, args.eval_step, args)

            #l_r.update(optimizer.param_groups[0]['lr'])
            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            #print(logits)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach_()/args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            mask1 = max_probs.ge(args.threshold1).float()
            
            #if ((epoch+1)>1):
            #targets_unl=targets_u*mask
            if ((epoch+1)>400 and (epoch+1)%2==0):
                ind_unl=i*mask1.cpu().detach()
                targets_unl=targets_u.cpu().detach().numpy()
                ind_unl=ind_unl.numpy()
                #targets_unl=targets_unl[np.nonzero(targets_unl)]
                ixss=np.nonzero(ind_unl)
                ind_unl=ind_unl[np.nonzero(ind_unl)]
                inds.append(ind_unl)
                targets_unl=targets_unl[ixss]
                lbs.append(targets_unl)
            
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            #if epoch>200:
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            mask1_probs.update(mask1.mean().item())
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. Mask1: {mask1:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg,
                    mask1=mask1_probs.avg))
                p_bar.update()
        
        if ((epoch+1)>400 and (epoch+1)%2==0):
        #if ((epoch+1)>20):  
            ixs=np.concatenate(inds).ravel()
            ls=np.concatenate(lbs).ravel()  
            ixs=ixs.astype(int)
            ls=ls.astype(int)  
            #print(ixs[:100])
            #print(ls.shape,ixs.shape)
            locs=np.nonzero(np.isin(ixs, unq_labeled_idx,invert=True))
            ixs=ixs[locs]
            ls=ls[locs]
            print(ixs.shape)
            #indcs = [np.where(a == ixs)[0] for a in np.unique(ixs)]
            #indcs=np.concatenate(indcs)
            ixs,indcs= np.unique(ixs, return_index=True)
            #print(indcs)
            #print(ixs)
            #print(ixs[:20])
            #ixs=ixs[indcs]
            #print(ixs[:100])
            #print(dup.shape)
            #print(dup[:20])
            print(ixs.shape)
            ls=ls[indcs]  
            #print(ls.shape)
            data_clean_ld=lol(args,unq_labeled_idx, unq_labeled_dataset,unq_unlabeled_trainloader, ixs, ls)   
            unq_unlabeled_dataset,unq_labeled_dataset,unq_labeled_idx,all_targets=label_denoising(args,all_targets, 10, data_clean_ld, unq_labeled_idx, labels,ixs,ls, epoch) 
            print(unq_labeled_dataset)
            print(unq_unlabeled_dataset)
            """
            labeled_trainloader = DataLoader(
                labeled_dataset,
                sampler=RandomSampler(labeled_dataset),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_last=True)
            """

            unq_labeled_trainloader = DataLoader(
                unq_labeled_dataset,
                sampler=SequentialSampler(unq_labeled_dataset),
                batch_size=16,
                num_workers=args.num_workers,
                drop_last=True)

            unq_unlabeled_trainloader = DataLoader(
                unq_unlabeled_dataset,
                sampler=SequentialSampler(unq_unlabeled_dataset),
                batch_size=16,
                num_workers=args.num_workers,
                drop_last=True)
            #for batch_idx,(inputs,targets, index) in enumerate (unq_labeled_trainloader):
            # print(index.item())

            
            #labeled_iter = iter(labeled_trainloader)
            #unlabeled_iter = iter(unlabeled_trainloader)
            #temp=round(mask1_probs.avg,3)
        
        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            writer.add_scalar('test/1.test_acc', test_acc, epoch)
            writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))
        

    if args.local_rank in [-1, 0]:
        writer.close()
def lol(args,unq_labeled_idx,unq_labeled_dataset,unq_unlabeled_trainloader,indexss, lb):
  cifar10_mean = (0.4914, 0.4822, 0.4465)
  cifar10_std = (0.2471, 0.2435, 0.2616)
  cifar100_mean = (0.5071, 0.4867, 0.4408)
  cifar100_std = (0.2675, 0.2565, 0.2761)
  normal_mean = (0.5, 0.5, 0.5)
  normal_std = (0.5, 0.5, 0.5)
  transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
  '''  
  indxs=[]
  lblslst=[]

  for batch_idx, (img, lbl, index) in enumerate(unq_unlabeled_trainloader):
        model.eval()
        img = img.to(args.device)
        out = model(img)   
        pseudo_label = torch.softmax(out.detach_()/args.T, dim=-1)
        max_probs, b = torch.max(pseudo_label, dim=-1)
        b=b.cpu().detach().numpy()                 # Generate predictions
        ind=index.numpy()
        indxs.append(ind)
        lblslst.append(b)
  lb=np.zeros(1)      
  indexss=np.zeros(1)
  for i in lblslst:
    lb=np.append(lb,i)
  lb=np.delete(lb,[0])

  for j in indxs:
    indexss=np.append(indexss,j)
  indexss=np.delete(indexss,[0])
  
    

  lb=lb.astype(int)
  indexss=indexss.astype(int)
  '''
  new_unq_unlbl_dataset = CIFAR10SSL1(
    "./data", indexss, lb, train=True,
    transform=transform_labeled)

  '''
        if batch_idx==0:
          a=img
          c=index
          img=img.to(args.device)
          logs=model(img)
          pseudo_label = torch.softmax(logs.detach_()/args.T, dim=-1)
          max_probs, b = torch.max(pseudo_label, dim=-1)
          b=b.detach().numpy
        else:
          a=torch.cat((a, img.detach()), 0)
          img=img.to(args.device)
          logs=model(img)
          pseudo_label = torch.softmax(logs.detach_()/args.T, dim=-1)
          max_probs, targ = torch.max(pseudo_label, dim=-1)
          b=torch.cat((b, targ, 0)
          c=torch.cat((c,index,0)
  '''
  my_dataset=ConcatDataset([unq_labeled_dataset,new_unq_unlbl_dataset])
  lab_idx=[]
  unlab_idx=[]
  lab_idx.extend(range(0,unq_labeled_idx.size))
  sz=len(my_dataset)
  unlab_idx.extend(range(unq_labeled_idx.size,sz))
  batch_sampler=TwoStreamBatchSampler(unlab_idx, lab_idx, 16, 8)
  my_dataloader = DataLoader(my_dataset,
          batch_sampler=batch_sampler,
          num_workers=args.num_workers)
          
  '''        
  #print(unq_labeled_dataset.data[0].ToTensor)
  #print(unq_labeled_dataset, new_unq_unlbl_dataset)  
  my_dataset=ConcatDataset([unq_labeled_dataset,new_unq_unlbl_dataset])
  my_dataloader = DataLoader(my_dataset,
          sampler=SequentialSampler(my_dataset),
          batch_size=16,
          num_workers=args.num_workers,
          drop_last=True) # create your dataloader
  '''        
  '''
  new_unq_unlbl_dataloader = DataLoader(new_unq_unlbl_dataset,
          sampler=SequentialSampler(new_unq_unlbl_dataset),
          batch_size=16,
          num_workers=args.num_workers,
          drop_last=True) # create your dataloader
  
  for batch_idx, (inputs, targets,index) in enumerate(my_dataloader):
    if batch_idx==0:
      print(inputs)
      
  '''
     
  return my_dataloader

def label_denoising(args,all_targets,itrs,loader,unq_labeled_idx,lbels,ix,lb,e):
    """
    all_embeddings = np.concatenate((support, query), axis=0)
    input_size = all_embeddings.shape[1]
    X = torch.tensor(all_embeddings, dtype=torch.float32, requires_grad=True)
    all_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    Y = torch.tensor(all_ys, dtype=torch.long)
    output_size = support_ys.max() + 1
    """

    start_lr = 0.15
    end_lr = 0.00
    cycle = 50 #number of epochs
    step_size_lr = (start_lr - end_lr) / cycle
   # print(input_size, output_size.item())
    lambda1 = lambda x: start_lr - (x % cycle)*step_size_lr
    o2u = o2u_()
    o2u=o2u.to(args.device)
    optimizer = optim.SGD(o2u.parameters(), 1, momentum=0.9, weight_decay=5e-4)
    scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    criterion = nn.CrossEntropyLoss(reduction='none')
    sz=len(loader)*8
    loss_statistics = np.zeros(sz)
    #loss_statistics=loss_statistics.to(args.device)
    lr_progression =[]
    o2u.train()
    
    #train_loss = AverageMeter()
    #l_each=[]
    #correct = 0
    total = 0
    indxs=[]
    lblslst=[]
    for epoch in range(itrs):
        l_each=[]
        for batch_idx, (inputs, targets, index) in enumerate(loader):
            inputs, targets= inputs.to(args.device), targets.to(args.device)
           # inputs, targets = inputs.cuda(), targets.cuda()
           # print(inputs.shape)
           # print(targets) 
            inputs, targets = Variable(inputs), Variable(targets)
            o2u.zero_grad()
            outputs = o2u(inputs)
            loss_each = criterion(outputs, targets)
            loss_all =  torch.mean(loss_each)
            loss_all.backward()
            #train_loss.update(loss_all.item())
            #loss_statistics = loss_statistics + loss_each/(itrs)
            optimizer.step()
            scheduler_lr.step()
            l_each.append(loss_each[:8].cpu().detach().numpy())
            lr_progression.append(optimizer.param_groups[0]['lr'])
            '''
            if epoch==0:
              b=targets.cpu().detach().numpy()                 # Generate predictions
              ind=index.numpy()
              indxs.append(ind)
              lblslst.append(b)
        ix=np.concatenate(indxs).ravel()
        lb=np.concatenate(lblslst).ravel()
        '''
        l_each=np.concatenate(l_each).ravel()
        loss_statistics=np.add(loss_statistics,l_each)
        #print(loss_statistics)
        '''
            #print(outputs)
            loss = criterion(outputs, targets)
            #loss =  torch.mean(loss_each)
            loss.backward()
            #l_each.append(loss_each.detach())
            optimizer.step()
            scheduler_lr.step()
           # print(loss_each.cpu())
            train_loss.update(loss.item())
        '''
    loss_statistics=np.divide(loss_statistics, float(itrs))
    #size=unq_labeled_idx.size
    #loss_statistics=loss_statistics[size:]
    ix_of_loss=np.argsort(loss_statistics)
    ix=ix[ix_of_loss]
    lb=lb[ix_of_loss]
    #print(ix[0:299])
    conf=[]
    confl=[]
    confident=ix[0]
    confident_target = lb[0]
    conf.append(confident)
    confl.append(confident_target)
    unq_labeled_idx=np.append(unq_labeled_idx,confident)
    all_targets=np.append(all_targets,confident_target)
    
    '''
    if (e+1)<=100:
      index_list=[]
      for i in range(10):
          count=0
          for j in range(len(lb)):
              if count<1:
                  if lb[j]==i:
                      index_list.append(j)
                      count+=1
              else:
                  break
    if (e+1)>100:
      index_list=[]
      for i in range(10):
          count=0
          for j in range(len(lb)):
              if count<2:
                  if lb[j]==i:
                      index_list.append(j)
                      count+=1
              else:
                  break
    '''
    '''
    conf=[]
    confl=[]
    index_list=[]
    for i in range(10):
      count=0
      for j in range(len(lb)):
          if count<2:
              if lb[j]==i:
                  index_list.append(j)
                  count+=1
          else:
              break
    for x in index_list:
        confident=ix[x]
        confident_target=lb[x]
        conf.append(confident)
        confl.append(confident_target)
        unq_labeled_idx=np.append(unq_labeled_idx,confident)
        all_targets=np.append(all_targets,confident_target)
    '''    
    confl=np.array(confl)
    conf=np.array(conf)
    print(unq_labeled_idx.shape)
    correct=(confl==lbels[conf])
    acc = correct.mean()
    with open('correct.npy', 'wb') as f:
        np.save(f, correct)
    print(correct)
    print(acc)
    
    '''
   # print(len(l_each))
    losss=[]
    indx=[]
    lbls=[]
    with torch.no_grad():
      for batch_ind,(images,labels,index) in enumerate(un_loader):
      #images, labels = batch 
          o2u.eval()
          lbl=labels.detach().numpy()
          images = images.to(args.device)
          labels = labels.to(args.device)
          out = o2u(images)                    # Generate predictions
          loss = F.cross_entropy(out, labels,reduce=False)
          los=loss.cpu().detach().numpy()
          losss.append(los)
          ind=index.detach().numpy()
          indx.append(ind)
          lbls.append(lbl)
   
    
    new_ind=np.zeros(1)      
    new_loss=np.zeros(1)
    for i in losss:
      new_loss=np.append(new_loss,i)
    new_loss=np.delete(new_loss,[0])
    
    for j in indx:
      new_ind=np.append(new_ind,j)
    new_ind=np.delete(new_ind,[0])
    new_lbls=np.zeros(1)
    for i in lbls:
      new_lbls=np.append(new_lbls,i)
    new_lbls=np.delete(new_lbls,[0])
    new_lbls=new_lbls.astype(int)
    new_ind=new_ind.astype(int)

    ix=np.argsort(new_loss)
    #print(ix[0:299])
    confident=new_ind[ix[0:300]]
    unq_labeled_idx=np.append(unq_labeled_idx,confident)
    
    confident_target = new_lbls[ix[0:300]]
    all_targets=np.append(all_targets,confident_target)
     
    #print(unq_labeled_idx.shape)
   
    #confident_dataset=TensorDataset()
    #unq_labeled_dataset=ConcatDataset[]

    #print(unq_labeled_idx.shape)
    '''
    train_unq_unlabeled_dataset,train_unq_labeled_dataset=relabel10(args,unq_labeled_idx,all_targets,lbels)
    return train_unq_unlabeled_dataset,train_unq_labeled_dataset,unq_labeled_idx,all_targets
    '''  
      hist1.flatten()
      idx.flatten()
      print(hist1)
      print(idx)
    '''
"""       
for epoch in range(opt.denoising_iterations):
    output = o2u(X)
    optimizer.zero_grad()
    loss_each = criterion(output, Y)
    loss_all = torch.mean(loss_each)
    loss_all.backward()
    loss_statistics = loss_statistics + loss_each/(opt.denoising_iterations)
    optimizer.step()
    scheduler_lr.step()
    lr_progression.append(optimizer.param_groups[0]['lr'])
return loss_statistics, lr_progression
"""
def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
