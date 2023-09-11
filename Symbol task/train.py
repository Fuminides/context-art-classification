from __future__ import division

import os
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


import numpy as np
import pandas as pd

import utils
#from model_gcn import GCN
from model_sym import SymModel 
from dataloader_sym import ArtDatasetSym



def save_model(args_dict, state, type='school', train_feature='kgm', append='gradient'):
    directory = args_dict.dir_model + "%s/"%(args_dict.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + train_feature + '_' + type + '_' + append + '_best_model.pth.tar'
    print('Model saved in ' + filename)
    torch.save(state, filename)


def resume(args_dict, model, optimizer):

    best_val = float(0)
    args_dict.start_epoch = 0
    if args_dict.resume:
        if os.path.isfile(args_dict.resume):
            print("=> loading checkpoint '{}'".format(args_dict.resume))
            checkpoint = torch.load(args_dict.resume)
            args_dict.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args_dict.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args_dict.resume))
            best_val = float(0)

    return best_val, model, optimizer


def trainEpoch(args_dict, train_loader, model, criterion, optimizer, epoch, symbol_task=False, final_epoch=False):

    # object to store & plot the losses
    losses = utils.AverageMeter()
    # switch to train mode
    model.train()    
   

    for batch_idx, (input, target, fiability) in enumerate(train_loader):

        # Inputs to Variable type
        if torch.cuda.is_available():
          input = input.to('cuda')

        target_var = list()
        for j in range(target.shape[1]):
          target_j = torch.tensor(target[:, j], dtype=torch.int64).to('cuda')
          if torch.cuda.is_available():
            target_j = target_j.to('cuda')
          target_var.append(torch.autograd.Variable(target_j))
        
        fiability = torch.tensor(fiability, dtype=torch.float32).to('cuda')

        output = model(input)
        denominator = 1 / len(target_var[j])
        train_loss = 0
        for j, target_var_j in enumerate(target_var):
          for target_symbol in range(target_var_j.shape[1]):
            train_loss += denominator * criterion(output, target_var_j[:, target_symbol]) # * fiability

        # Store loss
        losses.update(train_loss.item(), input.size(0))
        # Backpropagate loss and update weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        if epoch % 3 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))



def valEpoch(args_dict, val_loader, model, criterion, epoch, symbol_task=False):

    # switch to evaluation mode
    model.eval()
    full_fiability = []
    for batch_idx, (input, target, fiabilities) in enumerate(val_loader):
        # Inputs to Variable type
        if torch.cuda.is_available():
          input = input.to('cuda')

        target_var = list()
        for j in range(target.shape[1]):
          target_j = torch.tensor(target[:, j]).to('cuda')
          if torch.cuda.is_available():
            target_j = target_j.to('cuda')
          target_var.append(torch.autograd.Variable(target_j))
            
        output = model(input)
        
        pred = torch.argmax(output, dim=1)

        if batch_idx == 0:
            out = pred.cpu().numpy()
            label = torch.squeeze(target).cpu().numpy()
        else:
            out = np.concatenate((out, pred.cpu().numpy()), axis=0)
            label = np.concatenate((label, torch.squeeze(target).cpu().numpy()), axis=0)
        full_fiability.extend(fiabilities)

    conf_matrix = confusion_matrix(label, out)
    ponderated_succes = np.sum(np.equal(out, label) * np.array(full_fiability)) / np.sum(full_fiability)
    print('Ponderated accuracy: ' + str(ponderated_succes))
    print('Symbols detected {acc}'.format(acc=conf_matrix[0,0]))
    print('Absence detected {acc}'.format(acc=conf_matrix[1,1]))
    print(conf_matrix)

    return f1_score(label, out)


def train_symbol_classifier(args_dict):

    
    # Data transformation for training (with data augmentation) and validation
    train_transforms = transforms.Compose([
        transforms.Resize(256),  # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),  # we get only the center of that rescaled
        transforms.RandomCrop(224),  # random crop within the center crop (data augmentation)
        transforms.RandomHorizontalFlip(),  # random horizontal flip (data augmentation)
        transforms.ToTensor(),  # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),  # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),  # we get only the center of that rescaled
        transforms.ToTensor(),  # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        len(args_dict.targets)
    except:
        args_dict.targets = [args_dict.targets]

    # Dataloaders for training and validation
    semart_train_loader = ArtDatasetSym(args_dict, set='train', transform=train_transforms, symbol_detect=args_dict.targets)
    semart_val_loader = ArtDatasetSym(args_dict, set='val',  transform=val_transforms, symbol_detect=args_dict.targets)
    train_loader = torch.utils.data.DataLoader(
        semart_train_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Training loader with %d samples' % semart_train_loader.__len__())

    val_loader = torch.utils.data.DataLoader(
        semart_val_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Validation loader with %d samples' % semart_val_loader.__len__())
    

    # Define model
    if args_dict.targets is None:
        model = SymModel(semart_train_loader.symbol_context.shape[1], model=args_dict.architecture)
    else:
        model = SymModel(len(args_dict.targets), model=args_dict.architecture)

    if torch.cuda.is_available():
        model.cuda()

    # Loss and optimizer
    
    if len(args_dict.targets) > 1:
        class_loss = nn.BCEWithLogitsLoss()
        if torch.cuda.is_available():
            class_loss = class_loss.cuda()    
    else:
        
        class_loss = nn.CrossEntropyLoss()
        try:
            if torch.cuda.is_available():
                class_loss = class_loss.cuda()
        except AttributeError:
            pass

    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                                lr=args_dict.lr,
                                momentum=args_dict.momentum)

    # Resume training if needed
    best_val, model, optimizer = resume(args_dict, model, optimizer)


    # Now, let's start the training process!
    print('Start training Symbolic task...')
    pat_track = 0
    for epoch in range(args_dict.start_epoch, args_dict.nepochs):

        # Compute a training epoch
        trainEpoch(args_dict, train_loader, model, class_loss, optimizer, epoch, symbol_task=True)
        train_loader.dataset.generate_negative_samples()
        # Compute a validation epoch
        accval = valEpoch(args_dict, val_loader, model, class_loss, epoch, symbol_task=True)

        # check patience
        if accval <= best_val:
            pat_track += 1
        else:
            pat_track = 0
        if pat_track >= args_dict.patience:
            break

        # save if it is the best validation accuracy
        is_best = accval > best_val
        best_val = max(accval, best_val)
        if is_best:
            save_model(args_dict, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'valtrack': pat_track,
                'curr_val': accval,
            }, type=args_dict.att, train_feature=args_dict.embedds, append=args_dict.append)

        print('** Validation: %f (best f1 score) - %f (current f1 score) - %d (patience)' % (best_val, accval, pat_track))


def run_train(args_dict):

    # Set seed for reproducibility
    torch.manual_seed(args_dict.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args_dict.seed)

    train_symbol_classifier(args_dict)
    

