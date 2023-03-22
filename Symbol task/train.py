from __future__ import division

import os
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms


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


def extract_grad_cam_features(visual_model, data, target_var, args_dict, batch_idx, lenet_model):
    res_quant = np.zeros((data.shape[0], 4))
    res_size = np.zeros((data.shape[0], 2))
    for ix, image in enumerate(data):
        ix_0 = int(target_var[0][ix].cpu().numpy())
        ix_1 = int(target_var[1][ix].cpu().numpy())
        ix_2 = int(target_var[2][ix].cpu().numpy())
        ix_3 = int(target_var[3][ix].cpu().numpy())
        grad_cam_image = 0.25 * get_gradcam(visual_model, image, ix_0, 0) + \
                        0.25 * get_gradcam(visual_model, image, ix_1, 1) + \
                        0.25 * get_gradcam(visual_model, image, ix_2, 2) + \
                        0.25 * get_gradcam(visual_model, image, ix_3, 3)
        [quantity, size] = lenet_model(torch.unsqueeze(image, 0))

        res_quant[ix] = quantity.detach().cpu().numpy()
        res_size[ix] = size.detach().cpu().numpy()

    pd.DataFrame(res_quant).to_csv('./DeepFeatures/sym_grad_cam_train_quant_' + str(batch_idx) + '_' + str(args_dict.att) + '_' + str(args_dict.embedds) + '.csv')
    pd.DataFrame(res_size).to_csv('./DeepFeatures/sym_grad_cam_train_size_' + str(batch_idx) + '_' + str(args_dict.att) + '_' + str(args_dict.embedds) + '.csv')
    
    
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
   

    for batch_idx, (input, target) in enumerate(train_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            if torch.cuda.is_available():
                input_var.append(torch.autograd.Variable(input[j]).cuda())
            else:
                input_var.append(torch.autograd.Variable(input[j]))

            target_var = torch.tensor(np.array(target, dtype=np.float), dtype=torch.float32)

            if torch.cuda.is_available():
                target_var = target_var.cuda(non_blocking=True)
            

        
        output = model(input_var[0])
        train_loss = criterion(output, torch.squeeze(target_var))

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        if epoch % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))



def valEpoch(args_dict, val_loader, model, criterion, epoch, symbol_task=False):

    # switch to evaluation mode
    model.eval()

    acc_sample = 0
    symbols_detected = 0
    symbols_possible = 0
    acc_possible = 0
    absence_detected = 0
    absence_possible = 0

    for batch_idx, (input, target) in enumerate(val_loader):
        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            if torch.cuda.is_available():
                input_var.append(torch.autograd.Variable(input[j]).cuda())
            else:
                input_var.append(torch.autograd.Variable(input[j]))

        if not symbol_task:
        # Targets to Variable type
            target_var = list()
            for j in range(len(target)):
                target[j] = torch.tensor(np.array(target[j], dtype=np.uint8))

                if torch.cuda.is_available():
                    target[j] = target[j].cuda(non_blocking=True)

                target_var.append(torch.autograd.Variable(target[j]))
        else:
            target_var = torch.tensor(np.array(target, dtype=np.float))
            if torch.cuda.is_available():
                    target_var = target_var.cuda(non_blocking=True)

        output = model(input_var[0])
        
        pred = output > 0.5
        label_actual = torch.squeeze(target).cpu().numpy()
        symbols_detected += np.sum(np.logical_and(pred.cpu().numpy(), label_actual), axis=None) 
        symbols_possible += np.sum(label_actual, axis=None)
        acc_sample += np.sum(np.equal(pred.cpu().numpy(), label_actual), axis=None)
        try:
            acc_possible += pred.shape[0] * pred.shape[1]
        except IndexError:
            acc_possible += pred.shape[0]
            
        absence_detected += np.sum(np.logical_and(pred.cpu().numpy()<1, label_actual<1), axis=None)
        absence_possible += np.sum(np.logical_not(label_actual), axis=None)


    acc = acc_sample / acc_possible
    acc_symbols = symbols_detected / symbols_possible
    acc_absence = absence_detected / absence_possible

    print('Symbols detected {acc}'.format(acc=acc_symbols))
    print('Absence detected {acc}'.format(acc=acc_absence))

    
    # Print validation info
    print('Accuracy {acc}'.format(acc=acc))
    #plotter.plot('closs', 'val', 'Class Loss', epoch, losses.avg)
    #plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)

    # Return acc
    return acc


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
    semart_val_loader = ArtDatasetSym(args_dict, set='val',  transform=val_transforms, canon_list=semart_train_loader.symbols_names, symbol_detect=args_dict.targets)
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

        print('** Validation: %f (best acc) - %f (current acc) - %d (patience)' % (best_val, accval, pat_track))


def run_train(args_dict):

    # Set seed for reproducibility
    torch.manual_seed(args_dict.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args_dict.seed)

    train_symbol_classifier(args_dict)
    

