import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, unq_labeled_idx, unq_unlabeled_idx = x_u_split(
        args, base_dataset.targets)

    #all_indexs=np.concatenate((unq_labeled_idx, unq_unlabeled_idx), axis=0)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    
    train_unq_labeled_dataset = CIFAR10SSL(
        root, unq_labeled_idx, train=True,
        transform=transform_labeled)

    train_unq_unlabeled_dataset = CIFAR10SSL(
        root, unq_unlabeled_idx, train=True,
        transform=transform_labeled)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_unq_labeled_dataset, train_unq_unlabeled_dataset, unq_labeled_idx, unq_unlabeled_idx, np.array(base_dataset.targets)


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, unq_labeled_idx, unq_unlabeled_idx = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unq_labeled_dataset = CIFAR100SSL(
        root, unq_labeled_idx, train=True,
        transform=transform_labeled)

    train_unq_unlabeled_dataset = CIFAR100SSL(
        root, unq_unlabeled_idx, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_unq_labeled_dataset, train_unq_unlabeled_dataset, unq_labeled_idx, unq_unlabeled_idx, np.array(base_dataset.targets)


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    unq_labeled_idx=labeled_idx
    unq_unlabeled_idx=np.setdiff1d(unlabeled_idx, unq_labeled_idx)
    #assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    #print(unq_labeled_idx.shape)
    #print(unq_unlabeled_idx.shape)
    return labeled_idx, unlabeled_idx, unq_labeled_idx, unq_unlabeled_idx

def relabel10(args,unq_labeled_idx, all_targets, labels):
  transform_labeled = transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
          transforms.ToTensor(),
          transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
  with open('index.npy', 'wb') as f:
    np.save(f, unq_labeled_idx)
  with open('labels.npy', 'wb') as f1:  
    np.save(f1, all_targets)        
  unlabeled_idx = np.array(range(len(labels)))
  unq_unlabeled_idx=np.setdiff1d(unlabeled_idx, unq_labeled_idx)
  '''
  if args.expand_labels or len(unq_labeled_idx) < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / len(unq_labeled_idx))
        labeled_index = np.hstack([unq_labeled_idx for _ in range(num_expand_x)])
        n_labels=np.hstack([all_targets for _ in range(num_expand_x)])
  
  #np.random.shuffle(labeled_index)
  
  train_labeled_dataset = CIFAR10SSL1(
      "/home/infres/dpkontrazis/drazi/FixMatch-pytorch/data", labeled_index, n_labels, train=True,
      transform=transform_labeled)
  '''    
  train_unq_labeled_dataset = CIFAR10SSL1(
      "/content/CleanFixMatch/data", unq_labeled_idx, all_targets, train=True,
      transform=transform_labeled)
  train_unq_unlabeled_dataset = CIFAR10SSL(
      "/content/CleanFixmatch/data", unq_unlabeled_idx, train=True,
      transform=transform_labeled)
  return train_unq_unlabeled_dataset,train_unq_labeled_dataset

def relabel100(args,unq_labeled_idx, all_targets, labels):
  transform_labeled = transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
          transforms.ToTensor(),
          transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
  unlabeled_idx = np.array(range(len(labels)))
  unq_unlabeled_idx=np.setdiff1d(unlabeled_idx, unq_labeled_idx)
  if args.expand_labels or len(unq_labeled_idx) < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / len(unq_labeled_idx))
        labeled_index = np.hstack([unq_labeled_idx for _ in range(num_expand_x)])
        n_labels=np.hstack([all_targets for _ in range(num_expand_x)])
  #np.random.shuffle(labeled_index)
  
  train_labeled_dataset = CIFAR100SSL1(
      "/home/infres/dpkontrazis/drazi/FixMatch-pytorch/data", labeled_index, n_labels, train=True,
      transform=transform_labeled)
  train_unq_labeled_dataset = CIFAR100SSL1(
      "/home/infres/dpkontrazis/drazi/FixMatch-pytorch/data", unq_labeled_idx, all_targets, train=True,
      transform=transform_labeled)
  train_unq_unlabeled_dataset = CIFAR100SSL(
      "/home/infres/dpkontrazis/drazi/FixMatch-pytorch/data", unq_unlabeled_idx, train=True,
      transform=transform_labeled)
  return train_labeled_dataset,train_unq_unlabeled_dataset,train_unq_labeled_dataset

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

class CIFAR10SSL1(datasets.CIFAR10):
    def __init__(self, root, indexs, all_targets, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            #self.idx=self.data[index]
            self.data = self.data[indexs]
            self.targets = all_targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(index)
        return img, target, index


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            #self.idx=self.data[index]
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(index)
        return img, target, index

class CIFAR100SSL1(datasets.CIFAR100):
    def __init__(self, root, indexs, all_targets, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            #self.idx=self.data[index]
            self.data = self.data[indexs]
            self.targets = all_targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(index)
        return img, target, index

class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
