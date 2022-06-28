from __future__ import division
from operator import index

import os
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms

import numpy as np
import pandas as pd
from xgboost import train

import utils
#from model_gcn import GCN
from model_mtl import MTL
from model_sym import SymModel 
from model_kgm import KGM, KGM_append
from model_rmtl import RMTL
from dataloader_mtl import ArtDatasetMTL
from dataloader_kgm import ArtDatasetKGM
from dataloader_sym import ArtDatasetSym
from attributes import load_att_class

#from torch_geometric.loader import DataLoader
if torch.cuda.is_available():
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except ModuleNotFoundError:
        print('Pytorch geometric not found, proceeding...')
else:
    #from dataclasses import dataclass

    class Data(dict):
        def __init__(self, *args, **kwargs):
            super(Data, self).__init__(*args, **kwargs)
            self.__dict__ = self

def print_classes(type2idx, school2idx, timeframe2idx, author2idx):
    print('Att type\t %d classes' % len(type2idx))
    print('Att school\t %d classes' % len(school2idx))
    print('Att time\t %d classes' % len(timeframe2idx))
    print('Att author\t %d classes' % len(author2idx))


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


def trainEpoch(args_dict, train_loader, model, criterion, optimizer, epoch, symbol_task=None):

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

        if not symbol_task:
        # Targets to Variable type
            target_var = list()
            for j in range(len(target)):
                target[j] = torch.tensor(np.array(target[j], dtype=np.uint8))

                if torch.cuda.is_available():
                    target[j] = target[j].cuda(non_blocking=True)

                target_var.append(torch.autograd.Variable(target[j]))
        else:
            target_var = torch.tensor(np.array(target, dtype=np.float), dtype=torch.float32)
            if torch.cuda.is_available():
                    target_var = target_var.cuda(non_blocking=True)
            

        # Output of the model
        if args_dict.append == 'append':
            output = model((input_var[0], target[-1]))
        else:
            output = model(input_var[0])

        if args_dict.model != 'kgm':

            if args_dict.att == 'all':
                if args_dict.model == 'rmtl':
                    class_loss = multi_class_loss(criterion[0], target_var, output)
            
                    encoder_loss = criterion[1](output[4], output[5])
              
                    train_loss = args_dict.lambda_c * class_loss + \
                         args_dict.lambda_e * encoder_loss
                elif symbol_task:
                    train_loss = criterion(output, target_var)
                else: 
                    train_loss = multi_class_loss(criterion, target_var, output)
            else:
              if args_dict.model == 'rmtl':
                class_loss = criterion[0](output, target_var)
                encoder_loss = criterion[1](output[-2], output[-1])

                train_loss = args_dict.lambda_c * class_loss + \
                         args_dict.lambda_e * encoder_loss
              else:
                train_loss = criterion(output, target_var)

            losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # It is a Context-based model
        else:
            if args_dict.att == 'all': # TODO
                class_loss = multi_class_loss(criterion, target_var, output)
                
                encoder_loss = criterion[1](output[4], target_var[-1].long())
                train_loss = args_dict.lambda_c * class_loss + \
                            args_dict.lambda_e * encoder_loss

            else:
                if args_dict.append == 'append':
                    train_loss = criterion[0](output, target_var[0].long())
                    
                else:
                    class_loss = criterion[0](output[0], target_var[0].long())
                    encoder_loss = criterion[1](output[1], target_var[1].float())

                    train_loss = args_dict.lambda_c * class_loss + \
                                args_dict.lambda_e * encoder_loss

                
            losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        if epoch % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))

    # Plot
    #plotter.plot('closs', 'train', 'Class Loss', epoch, losses.avg)


def valEpoch(args_dict, val_loader, model, criterion, epoch, symbol_task=False):

    # switch to evaluation mode
    model.eval()
    acc_sample = 0
    acc_possible = 0

    for batch_idx, (input, target) in enumerate(val_loader):
        print(target.shape)
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

        # Predictions
        with torch.no_grad():
            # Output of the model
            if args_dict.append == 'append':
                output = model((input_var[0], target[1]))
            else:
                output = model(input_var[0])
        if symbol_task:
            pred = output > 0.5
            label_actual = target.cpu().numpy()

            print(pred.shape, label_actual.shape)
            acc_sample += np.equal(pred.cpu().numpy(), label_actual)
            acc_possible += pred.shape[0] * pred.shape[1]

        elif args_dict.att == 'all':
            pred_type = torch.argmax(output[0], 1)
            pred_school = torch.argmax(output[1], 1)
            pred_time = torch.argmax(output[2], 1)
            pred_author = torch.argmax(output[3], 1)

            # Save predictions to compute accuracy
            if batch_idx == 0:
                out_type = pred_type.data.cpu().numpy()
                out_school = pred_school.data.cpu().numpy()
                out_time = pred_time.data.cpu().numpy()
                out_author = pred_author.data.cpu().numpy()
                label_type = target[0].cpu().numpy()
                label_school = target[1].cpu().numpy()
                label_tf = target[2].cpu().numpy()
                label_author = target[3].cpu().numpy()
            else:
                out_type = np.concatenate((out_type, pred_type.data.cpu().numpy()), axis=0)
                out_school = np.concatenate((out_school, pred_school.data.cpu().numpy()), axis=0)
                out_time = np.concatenate((out_time, pred_time.data.cpu().numpy()), axis=0)
                out_author = np.concatenate((out_author, pred_author.data.cpu().numpy()), axis=0)
                label_type = np.concatenate((label_type, target[0].cpu().numpy()), axis=0)
                label_school = np.concatenate((label_school, target[1].cpu().numpy()), axis=0)
                label_tf = np.concatenate((label_tf, target[2].cpu().numpy()), axis=0)
                label_author = np.concatenate((label_author, target[3].cpu().numpy()), axis=0)
           
        else:
            if args_dict.model == 'kgm' and (not args_dict.append == 'append'):
                pred = torch.argmax(output[0], 1)
                label_actual = target[0].cpu().numpy()
                
                # Save predictions to compute accuracy
                if batch_idx == 0:
                    out = pred.data.cpu().numpy()
                    label = label_actual
                    
                else:
                    out = np.concatenate((out, pred.data.cpu().numpy()), axis=0)
                    label = np.concatenate((label, label_actual), axis=0)
                    
            elif args_dict.model == 'kgm':
                pred = torch.argmax(output, 1)
                label_actual = target[0].cpu().numpy()
            else:
                pred = torch.argmax(output, 1)
                label_actual = target.cpu().numpy()

             
            # Save predictions to compute accuracy
            if batch_idx == 0:
                out = pred.data.cpu().numpy()
                label = label_actual
                
            else:
                out = np.concatenate((out, pred.data.cpu().numpy()), axis=0)
                label = np.concatenate((label, label_actual), axis=0)
                
    # Accuracy
    if symbol_task:
        acc = acc_sample / acc_possible
    elif args_dict.att == 'all':
        acc_type = np.sum(out_type == label_type)/len(out_type)
        acc_school = np.sum(out_school == label_school) / len(out_school)
        acc_tf = np.sum(out_time == label_tf) / len(out_time)
        acc_author = np.sum(out_author == label_author) / len(out_author)
        acc = np.mean((acc_type, acc_school, acc_tf, acc_author))

    elif args_dict.model == 'kgm' or symbol_task:
        acc = np.sum(out == label) / len(out)

    # Print validation info
    print('Accuracy {acc}'.format(acc=acc))
    #plotter.plot('closs', 'val', 'Class Loss', epoch, losses.avg)
    #plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)

    # Return acc
    return acc

def multi_class_loss(criterion, target_var, output):
    return 0.25 * criterion(output[0], target_var[0].long()) + \
                            0.25 * criterion(output[1], target_var[1].long()) + \
                            0.25 * criterion(output[2], target_var[2].long()) + \
                            0.25 * criterion(output[3], target_var[3].long())


def train_knowledgegraph_classifier(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    if args_dict.att == 'type':
        att2i = type2idx
    elif args_dict.att == 'school':
        att2i = school2idx
    elif args_dict.att == 'time':
        att2i = time2idx
    elif args_dict.att == 'author':
        att2i = author2idx
    N_CLUSTERS = args_dict.clusters

    # Define model
    if args_dict.embedds == 'graph':
        if args_dict.append == 'append':
            model = KGM(len(att2i), end_dim=N_CLUSTERS)
        else:
            model = KGM_append(len(att2i), end_dim=N_CLUSTERS)
    else:
        if args_dict.append != 'append':
            model = KGM(len(att2i), end_dim=N_CLUSTERS)
        else:
            model = KGM_append(len(att2i), end_dim=N_CLUSTERS)

    if torch.cuda.is_available():#args_dict.use_gpu:
        model.cuda()

    # Loss and optimizer
    if torch.cuda.is_available():
        class_loss = nn.CrossEntropyLoss().cuda()
    else:
        class_loss = nn.CrossEntropyLoss()

    encoder_loss = nn.SmoothL1Loss()
    loss = [class_loss, encoder_loss]
    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=args_dict.lr, momentum=args_dict.momentum)

    # Resume training if needed
    best_val, model, optimizer = resume(args_dict, model, optimizer)

    # Data transformation for training (with data augmentation) and validation
    train_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.RandomHorizontalFlip(),                  # random horizontal flip (data augmentation)
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])
    k = args_dict.k

    # Dataloaders for training and validation
    semart_train_loader = ArtDatasetKGM(args_dict, set='train', att2i=att2i, att_name=args_dict.att, append=args_dict.append, transform=train_transforms, clusters=N_CLUSTERS, k=k)
    semart_val_loader = ArtDatasetKGM(args_dict, set='val', att2i=att2i, att_name=args_dict.att, transform=val_transforms, clusters=N_CLUSTERS, k=k)

    train_loader = torch.utils.data.DataLoader(
        semart_train_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Training loader with %d samples' % semart_train_loader.__len__())

    val_loader = torch.utils.data.DataLoader(
        semart_val_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Validation loader with %d samples' % semart_val_loader.__len__())

    # Now, let's start the training process!
    print('Start training KGM model...')
    pat_track = 0
    for epoch in range(args_dict.start_epoch, args_dict.nepochs):

        # Compute a training epoch
        trainEpoch(args_dict, train_loader, model, loss, optimizer, epoch, symbol_task=args_dict.symbol_task)

        # Compute a validation epoch
        accval = valEpoch(args_dict, val_loader, model, loss, epoch, symbol_task=args_dict.symbol_task)

        # check patience
        if accval <= best_val:
            pat_track += 1
        else:
            pat_track = 0
        if pat_track >= args_dict.patience:
            break

        # save if it is the best model
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


def train_multitask_classifier(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]

    # Define model
    model = MTL(num_classes, model=args_dict.architecture)
    if torch.cuda.is_available():
        model.cuda()

    # Loss and optimizer
    if torch.cuda.is_available():
        class_loss = nn.CrossEntropyLoss().cuda()
    else:
        class_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                                lr=args_dict.lr,
                                momentum=args_dict.momentum)

    # Resume training if needed
    best_val, model, optimizer = resume(args_dict, model, optimizer)

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


    # Dataloaders for training and validation
    semart_train_loader = ArtDatasetMTL(args_dict, set='train', att2i=att2i, transform=train_transforms)
    semart_val_loader = ArtDatasetMTL(args_dict, set='val', att2i=att2i, transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(
        semart_train_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Training loader with %d samples' % semart_train_loader.__len__())

    val_loader = torch.utils.data.DataLoader(
        semart_val_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Validation loader with %d samples' % semart_val_loader.__len__())

    # Now, let's start the training process!
    print_classes(type2idx, school2idx, time2idx, author2idx)
    print('Start training MTL model...')
    pat_track = 0
    for epoch in range(args_dict.start_epoch, args_dict.nepochs):

        # Compute a training epoch
        trainEpoch(args_dict, train_loader, model, class_loss, optimizer, epoch)

        # Compute a validation epoch
        accval = valEpoch(args_dict, val_loader, model, class_loss, epoch)

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


    # Dataloaders for training and validation
    semart_train_loader = ArtDatasetSym(args_dict, set='train', transform=train_transforms)
    semart_val_loader = ArtDatasetSym(args_dict, set='val',  transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(
        semart_train_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Training loader with %d samples' % semart_train_loader.__len__())

    val_loader = torch.utils.data.DataLoader(
        semart_val_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Validation loader with %d samples' % semart_val_loader.__len__())
    

    # Define model
    model = SymModel(len(semart_train_loader.symbols_names), model=args_dict.architecture)
    if torch.cuda.is_available():
        model.cuda()

    # Loss and optimizer
    if torch.cuda.is_available():
        class_loss = nn.CrossEntropyLoss().cuda()
    else:
        class_loss = nn.CrossEntropyLoss()

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


def gen_embeds(args_dict, vis_encoder, data_partition='train'):
    '''
    Generates the proper dataset to train the GCN model.
        1. A file containing the embeddings for each category and painting for train, validation and test.
        
        
        NOTE: remember that the pseudo-labels for validation and test are computed
        using another classification model.
    '''
    from torchvision import transforms

    transforms = transforms.Compose([
        transforms.Resize(256),  # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),  # we get only the center of that rescaled
        transforms.RandomCrop(224),  # random crop within the center crop (data augmentation)
        transforms.RandomHorizontalFlip(),  # random horizontal flip (data augmentation)
        transforms.ToTensor(),  # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])
    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]

    from PIL import Image
    from model_rmtl import RMTL

    #if args_dict.embedds == 'graph':
    if data_partition == 'train':
        train_node2vec_emb = pd.read_csv('Data/semart.emd', skiprows=1, sep=' ', header=None, index_col=0)
    else:
        train_node2vec_emb = pd.read_csv('Data/semart_' + data_partition + '.emd', skiprows=1, sep=' ', header=None, index_col=0)


    vis_encoder.eval()
    if torch.cuda.is_available():
        vis_encoder = vis_encoder.cuda()

    # feature_matrix = np.zeros(train_node2vec_emb.shape)
    print('Starting the process... ')
    semart_train_loader = ArtDatasetMTL(args_dict, set=data_partition, att2i=att2i, transform=transforms)
    train_loader = torch.utils.data.DataLoader(
        semart_train_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)

    type_label = []
    school_label = []
    time_label = []
    author_label = []

    for batch_idx, (input, target) in enumerate(train_loader):

        # Inputs to Variable type
        input_var = list()
        type_label.append(target[0])
        school_label.append(target[0])
        time_label.append(target[0])
        author_label.append(target[0])

        for j in range(len(input)):
            if torch.cuda.is_available():
                input_var.append(torch.autograd.Variable(input[j]).cuda())
            else:
                input_var.append(torch.autograd.Variable(input[j]))

        if batch_idx == 0:
            features_matrix = vis_encoder.reduce(input_var[0])
            features_matrix = features_matrix.data.cpu().numpy()
        else:
            features_matrix = np.append(features_matrix, vis_encoder.reduce(input_var[0]).data.cpu().numpy(), axis=0)

        print(
            'Sample ' + str(batch_idx * int(args_dict.batch_size)) + 'th out of ' + str(len(semart_train_loader)))

    #if args_dict.embedds == 'graph':
    features_matrix_end = np.append(features_matrix, train_node2vec_emb[len(semart_train_loader):], axis=0)
    '''elif args_dict.embedds == 'avg':
        # Gen the feature for each category using the avg of all their representatives
        additional_entities = np.zeros((train_node2vec_emb - len(semart_train_loader), train_node2vec_emb.shape[1]))
        for entity in range(additional_entities.shape[0]):
            entity
        features_matrix_end = np.append(features_matrix, train_node2vec_emb[len(semart_train_loader):], axis=0)'''
    
    assert train_node2vec_emb.shape == features_matrix_end.shape
    
    return features_matrix_end

def vis_encoder_train(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]

    # Define model
    model = RMTL(num_classes)
    if torch.cuda.is_available():
        model.cuda()

    # Loss and optimizer
    if torch.cuda.is_available():
        class_loss = nn.CrossEntropyLoss().cuda()
    else:
        class_loss = nn.CrossEntropyLoss()

    encoder_loss = nn.SmoothL1Loss()
    loss = [class_loss, encoder_loss]

    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                                lr=args_dict.lr,
                                momentum=args_dict.momentum)

    # Resume training if needed
    best_val, model, optimizer = resume(args_dict, model, optimizer)

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


    # Dataloaders for training and validation
    semart_train_loader = ArtDatasetMTL(args_dict, set='train', att2i=att2i, transform=train_transforms)
    semart_val_loader = ArtDatasetMTL(args_dict, set='val', att2i=att2i, transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(
        semart_train_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Training loader with %d samples' % semart_train_loader.__len__())

    val_loader = torch.utils.data.DataLoader(
        semart_val_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    print('Validation loader with %d samples' % semart_val_loader.__len__())

    # Now, let's start the training process!
    print_classes(type2idx, school2idx, time2idx, author2idx)
    print('Start training RMTL model...')
    pat_track = 0
    for epoch in range(args_dict.start_epoch, args_dict.nepochs):

        # Compute a training epoch
        trainEpoch(args_dict, train_loader, model, loss, optimizer, epoch)

        # Compute a validation epoch
        accval = valEpoch(args_dict, val_loader, model, loss, epoch)

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

            feature_matrix = gen_embeds(args_dict, model, 'train')
            print(feature_matrix[0:2, :][:10])
            pd.DataFrame(feature_matrix).to_csv('Data/feature_train_128_semart.csv')

            feature_matrix = gen_embeds(args_dict, model, 'val')
            pd.DataFrame(feature_matrix).to_csv('Data/feature_val_128_semart.csv')

            feature_matrix = gen_embeds(args_dict, model, 'test')
            pd.DataFrame(feature_matrix).to_csv('Data/feature_test_128_semart.csv')


        print('** Validation: %f (best acc) - %f (current acc) - %d (patience)' % (best_val, accval, pat_track))

def _load_labels(df_path, att2i):
    def class_from_name(name, vocab):

        if name in vocab:
            idclass= vocab[name]
        else:
            idclass = vocab['UNK']

        return int(idclass)

    df = pd.read_csv(df_path, delimiter='\t', encoding='Cp1252')
    
    type_vocab = att2i[0]
    school_vocab = att2i[1]
    time_vocab = att2i[2]
    author_vocab = att2i[3]
    
    tipe = list(df['TYPE'])
    school = list(df['SCHOOL'])
    time = list(df['TIMEFRAME'])
    author = list(df['AUTHOR'])
    
    tipei = np.array([class_from_name(x, type_vocab) for x in tipe])
    schooli = np.array([class_from_name(x, school_vocab) for x in school])
    timei = np.array([class_from_name(x, time_vocab) for x in time])
    authori = np.array([class_from_name(x, author_vocab) for x in author])
    
    return tipei, schooli, timei, authori
    

def train_gcn_classifier(args_dict):    
    from model_gcn import GCN, NODE2VEC_OUTPUT
    from model_gat import GAT
    from torch_geometric.loader import DataLoader, NeighborSampler
    

    target = 'time'
    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]
    
    # Dataloaders for training and validation
    target_var_train = _load_labels(args_dict.dir_dataset + '/semart_train.csv', att2i)
    target_var_val = _load_labels(args_dict.dir_dataset + '/semart_val.csv', att2i)
    target_var_test = _load_labels(args_dict.dir_dataset + '/semart_test.csv', att2i)
    og_train_size = len(target_var_train[0])
    val_size = len(target_var_val[0])

    data = load_gcn_data(args_dict, og_train_size, val_size)
    loader = NeighborSampler(
        data.edge_index, node_idx=data.train_mask,#+data.val_mask,
        sizes=[10, 5], batch_size=int(args_dict.batch_size), shuffle=True, num_workers=0)
    
    '''if torch.cuda.is_available():
        train_edge_list = torch.tensor(np.array(train_edge_list).reshape(2, train_edge_list.shape[0])).cuda()
        val_edge_list = torch.tensor(np.array(val_edge_list).reshape(2, val_edge_list.shape[0])).cuda()
        tensor_val_edge_list = tensor_val_edge_list.cuda()
        train_mask = train_mask.cuda()'''

    # Define model
    if args_dict.model == 'gcn':
      model = GCN(NODE2VEC_OUTPUT, 16, num_classes,target_class=target)
    elif args_dict.model == 'gat':
      model = GAT(NODE2VEC_OUTPUT, 16, num_classes,target_class=target)

    if torch.cuda.is_available():
        model.cuda()

    # Loss and optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                                lr=args_dict.lr,
                                momentum=args_dict.momentum)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

    # Resume training if needed
    best_val, model, optimizer = resume(args_dict, model, optimizer)

    column_key = {'type':0, 'school':1, 'time':2, 'author':3}

    # Now, let's start the training process!
    print_classes(type2idx, school2idx, time2idx, author2idx)

    print('Start training GCN model...')
    pat_track = 0
    for epoch in range(args_dict.start_epoch, args_dict.nepochs):
        print(epoch)

        for batch in loader:
        # Targets to Variable type
            batch_size, n_id, adjs = batch
            #if torch.cuda.is_available():
                #adjs = [adj.cuda() for adj in adjs]

            # edge_index, e_id, size = adjs[1]

            target_var = list()
            for j in range(len(target_var_train)):
                if torch.cuda.is_available():
                    aux = torch.tensor(target_var_train[j]).cuda(non_blocking=True)
                else:
                    aux = torch.tensor(target_var_train[j])

                target_var.append(torch.autograd.Variable(aux))

            # Compute a training epoch
            optimizer.zero_grad()

            output = model(data.x[n_id], adjs)
            index_loss_bool = n_id < og_train_size
            index_loss = index_loss[index_loss_bool]
            if target == 'all':
                train_loss = multi_class_loss(criterion, target_var, output, index_loss)
            else:
                train_loss = criterion(output[index_loss], target_var[column_key[target]][index_loss])
            #print(train_loss)
            train_loss.backward()
            optimizer.step()
            #scheduler.step()

        print('************')
        if target == 'all' or target == 'type':
          acc_type = np.mean(np.equal(torch.argmax(output[0][0:og_train_size]).data.cpu().numpy(), target_var_train[0]))
          print('Train Type: ' + str(acc_type))

        elif target == 'all' or target == 'school':
          acc_school = np.mean(np.equal(torch.argmax(output[1][0:og_train_size]).data.cpu().numpy(), target_var_train[1]))
          print('Train School: ' + str(acc_school))

        elif target == 'all' or target == 'time':
          acc_tf = np.mean(np.equal(torch.argmax(output[2][0:og_train_size]).data.cpu().numpy(), target_var_train[2])) 
          print('Train TimeFrame: ' + str(acc_tf))

        elif target == 'all' or target == 'author':
          acc_author = np.mean(np.equal(torch.argmax(output[3][0:og_train_size]).data.cpu().numpy(), target_var_train[3])) 
          print('Train Author: ' + str(acc_author))

        if target == 'all':
          accval = np.mean((acc_type, acc_school, acc_tf, acc_author))
          print('Train: ' + str(accval))
        print('************')
        
        # Compute a validation epoch
        label_type = target_var_val[0]#.cpu().numpy()
        label_school = target_var_val[1]#.cpu().numpy()
        label_tf = target_var_val[2]#.cpu().numpy()
        label_author = target_var_val[3]#.cpu().numpy()

        # accval = valEpoch(args_dict, val_loader, model, class_loss, epoch)
        output = model(data.x[data.val_mask], data.val_edge_index)
        if target == 'all' or target == 'type':
          pred_type = torch.argmax(output[0], 1) if target == 'all' else torch.argmax(output, 1)
          out_type = pred_type.data.cpu().numpy()[-val_size:]
          acc_type = np.mean(np.equal(out_type, label_type))
          accval = acc_type
        elif target == 'all' or target == 'school':
          pred_school = torch.argmax(output[1], 1) if target == 'all' else torch.argmax(output, 1)
          out_school = pred_school.data.cpu().numpy()[-val_size:]
          acc_school = np.mean(np.equal(out_school, label_school)) 
          accval = acc_school
        elif target == 'all' or target == 'time':
          pred_time = torch.argmax(output[2], 1) if target == 'all' else torch.argmax(output, 1)
          out_time = pred_time.data.cpu().numpy()[-val_size:]
          acc_tf = np.mean(np.equal(out_time, label_tf))
          accval = acc_tf
        elif target == 'all' or target == 'author':  
          pred_author = torch.argmax(output[3], 1) if target == 'all' else torch.argmax(output, 1)
          out_author = pred_author.data.cpu().numpy()[-val_size:]
          acc_author = np.mean(np.equal(out_author, label_author))
          accval = acc_author

        if target == 'all':
          accval = np.mean((acc_type, acc_school, acc_tf, acc_author))
        output = model(data.x, data.val_edge_index)
        accval = compute_preds_val(target, val_size, output, label_type, label_school, label_tf, label_author)

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

def load_gcn_data(args_dict, og_train_size, val_size):
    # Load semart knowledge graphs
    train_edge_list = pd.read_csv(args_dict.edge_list_train, index_col=None, sep=' ', header=None)
    val_edge_list = pd.read_csv(args_dict.edge_list_val, index_col=None, sep=' ', header=None)
    val_edge_list = pd.concat([train_edge_list, val_edge_list], axis=0)

    tensor_train_edge_list = torch.tensor(np.array(train_edge_list).reshape((2, train_edge_list.shape[0])), dtype=torch.long)
    tensor_val_edge_list = torch.tensor(np.array(val_edge_list).reshape((2, val_edge_list.shape[0])), dtype=torch.long)
    
    # Load the feature matrix from the vis+node2vec representations
    train_feature_matrix = pd.read_csv(args_dict.feature_matrix, sep=',', header=None, skiprows=1, index_col=0)
    train_size = train_feature_matrix.shape[0]
    val_feature_matrix = pd.read_csv(args_dict.val_feature_matrix, sep=',',  header=None, skiprows=1, index_col=0)
    
    #torch_train_feature_matrix = torch.tensor(train_feature_matrix)
    total_samples = torch.tensor(np.array(pd.concat([train_feature_matrix, val_feature_matrix], axis=0))).float()
    n_samples = total_samples.shape[0]

    # Gen the train/val/test indexes
    train_mask = np.array([0] * n_samples)
    train_mask[0:train_size] = 1
    train_mask = torch.tensor(train_mask, dtype=torch.uint8)

    og_train_mask = np.array([0] * n_samples)
    og_train_mask[0:og_train_size] = 1
    og_train_mask = torch.tensor(og_train_mask, dtype=torch.uint8)

    val_mask = np.array([0] * n_samples)
    val_mask[train_size:train_size+val_size] = 1
    val_mask = torch.tensor(val_mask, dtype=torch.uint8)
    val_mask = torch.logical_or(train_mask, val_mask)

    test_mask = np.array([0] * n_samples)
    test_mask[train_size + val_size:] = 1
    test_mask = torch.tensor(test_mask, dtype=torch.uint8)

    if torch.cuda.is_available():
        tensor_train_edge_list = tensor_train_edge_list.cuda()
        tensor_val_edge_list = tensor_val_edge_list.cuda()
        total_samples = total_samples.cuda()

    #Load all the data as Data object for pytorch geometric
    data = Data(x=total_samples, edge_index=tensor_train_edge_list, val_edge_index=tensor_val_edge_list)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.og_train_mask = og_train_mask

    return data

def compute_preds_val(target, val_size, output, label_type, label_school, label_tf, label_author):
    if target == 'all' or target == 'type':
      _, pred_type = torch.argmax(output[0], 1) if target == 'all' else torch.max(output, 1)
      out_type = pred_type.data.cpu().numpy()[-val_size:]
      acc_type = np.sum(np.equal(out_type, label_type))/len(out_type)
      accval = acc_type
    elif target == 'all' or target == 'school':
      _, pred_school = torch.argmax(output[1], 1) if target == 'all' else torch.max(output, 1)
      out_school = pred_school.data.cpu().numpy()[-val_size:]
      acc_school = np.sum(np.equal(out_school, label_school)) / len(out_school)
      accval = acc_school
    elif target == 'all' or target == 'time':
      _, pred_time = torch.argmax(output[2], 1) if target == 'all' else torch.max(output, 1)
      out_time = pred_time.data.cpu().numpy()[-val_size:]
      acc_tf = np.sum(np.equal(out_time, label_tf)) / len(out_time)
      accval = acc_tf
    elif target == 'all' or target == 'author':  
      _, pred_author = torch.argmax(output[3], 1) if target == 'all' else torch.max(output, 1)
      out_author = pred_author.data.cpu().numpy()[-val_size:]
      acc_author = np.sum(np.equal(out_author, label_author)) / len(out_author)
      accval = acc_author

    if target == 'all':
      accval = np.mean((acc_type, acc_school, acc_tf, acc_author))

    return accval


def run_train(args_dict):

    # Set seed for reproducibility
    torch.manual_seed(args_dict.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args_dict.seed)

    # Plots
    #global plotter
    #plotter = utils.VisdomLinePlotter(env_name=args_dict.name)

    if args_dict.symbol_task:
        train_symbol_classifier(args_dict)
    elif args_dict.model == 'mtl':
        train_multitask_classifier(args_dict)
    elif args_dict.model == 'kgm':
        train_knowledgegraph_classifier(args_dict)
    elif args_dict.model == 'rmtl':
        vis_encoder_train(args_dict)
    elif args_dict.model == 'gcn':
        train_gcn_classifier(args_dict)
    elif args_dict.model == 'gat':
        train_gcn_classifier(args_dict)
    else:
        assert False, 'Incorrect model type'

