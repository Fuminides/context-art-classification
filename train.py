from __future__ import division
from operator import index

import os
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
import torchvision.ops

import numpy as np
import pandas as pd
from xgboost import train

import utils
#from model_gcn import GCN
from model_mtl import MTL
from model_kgm import KGM, KGM_append, GradCamKGM, get_gradcam
from dataloader_mtl import ArtDatasetMTL
from dataloader_kgm import ArtDatasetKGM
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
    filename = directory + args_dict.resume
    print('Model saved in ' + filename)
    torch.save(state, filename)


def extract_grad_cam_features(visual_model, data, target_var, args_dict, batch_idx, im_names, set_data='train'):
    for ix, image in enumerate(data):
        ix_0 = int(target_var[0][ix].cpu().numpy())
        ix_1 = int(target_var[1][ix].cpu().numpy())
        ix_2 = int(target_var[2][ix].cpu().numpy())
        ix_3 = int(target_var[3][ix].cpu().numpy())
        grad_cam_image = 0.25 * get_gradcam(visual_model, image, ix_0, 0) + \
                        0.25 * get_gradcam(visual_model, image, ix_1, 1) + \
                        0.25 * get_gradcam(visual_model, image, ix_2, 2) + \
                        0.25 * get_gradcam(visual_model, image, ix_3, 3)
        
        if ix == 0:
            grad_cams = torch.zeros((data.shape[0], 1, grad_cam_image.shape[0], grad_cam_image.shape[1]))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            grad_cams = grad_cams.to(device)

        grad_cams[ix] = grad_cam_image

    for jx in range(grad_cams.shape[0]):
        grad_cam = grad_cams[jx, 0, :, :].detach().cpu().numpy()
        pd.DataFrame(grad_cam).to_csv('./GradCams/' + im_names[jx] + '.csv', index=False)



   
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
    mtl_mode = args_dict.att == 'all'
    # switch to train mode
    model.train()
    actual_index = 0
    grad_classifier_path = args_dict.grad_cam_model_path
    
    grad_cam = True
    
    for batch_idx, (input, target, im_names) in enumerate(train_loader):

        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            if torch.cuda.is_available():
                input_var.append(torch.autograd.Variable(input[j]).cuda())
            else:
                input_var.append(torch.autograd.Variable(input[j]))

            # Targets to Variable type
            target_var = list()
            for j in range(len(target)):
                target[j] = torch.tensor(np.array(target[j], dtype=np.uint8))

                if torch.cuda.is_available():
                    target[j] = target[j].cuda(non_blocking=True)

                target_var.append(torch.autograd.Variable(target[j]))
                    

        # Output of the model
        if args_dict.append == 'append':
            output = model((input_var[0], target[-1]))
        else:
            output = model(input_var[0])

       
        if args_dict.model != 'kgm':

            if args_dict.att == 'all':
                    train_loss = multi_class_loss(criterion, target_var, output)
            else:
                train_loss = criterion(output, torch.squeeze(target_var))

            losses.update(train_loss.data.cpu().numpy(), input[0].size(0))
        
        # It is a Context-based model
        else:

            '''if final_epoch:
                print('Saving features...')
                feat = model.features(input_var[0])

                pd.DataFrame(feat.data.cpu().numpy(), index=im_names).to_csv('./DeepFeatures/train_x_' + str(batch_idx) + '_' + str(args_dict.att) + '_' + str(args_dict.embedds) + '.csv', index=True)
                if not mtl_mode:
                    pd.DataFrame(target_var[0].cpu().numpy()).to_csv('./DeepFeatures/train_y_' + str(batch_idx) + '_' + str(args_dict.att) + '_' + str(args_dict.embedds) + '.csv')
                else:
                    if grad_cam:
                        extract_grad_cam_features(model, input_var[0], target_var, args_dict, batch_idx, im_names, set_data='train')
                    # pd.DataFrame(target_var[0].cpu().numpy()).to_csv('./DeepFeatures/train_y_' + str(batch_idx) + '_' + str('type') + '_' + str(args_dict.embedds) + '.csv')
                    # pd.DataFrame(target_var[1].cpu().numpy()).to_csv('./DeepFeatures/train_y_' + str(batch_idx) + '_' + str('school') + '_' + str(args_dict.embedds) + '.csv')
                    # pd.DataFrame(target_var[2].cpu().numpy()).to_csv('./DeepFeatures/train_y_' + str(batch_idx) + '_' + str('timeframe') + '_' + str(args_dict.embedds) + '.csv')
                    # pd.DataFrame(target_var[3].cpu().numpy()).to_csv('./DeepFeatures/train_y_' + str(batch_idx) + '_' + str('author') + '_' + str(args_dict.embedds) + '.csv')
            '''
            actual_index += args_dict.batch_size
            
            if args_dict.att == 'all':
                class_loss = multi_class_loss(criterion[0], target_var, output)
                
                encoder_loss = criterion[1](output[4], target_var[-1].float())
                train_loss = args_dict.lambda_c * class_loss + \
                            args_dict.lambda_e * encoder_loss

            else:
                if args_dict.append == 'append':
                    train_loss = criterion[0](output, target_var[0].long())
                    
                else:
                    class_loss = criterion[0](output[0], target_var[0].long())
                    encoder_loss = criterion[1](output[1], target_var[1].float())

                    if args_dict.base == 'base':
                        train_loss = class_loss
                    else:
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
    symbols_detected = 0
    symbols_possible = 0
    acc_possible = 0
    absence_detected = 0
    absence_possible = 0

    for batch_idx, (input, target, im_names) in enumerate(val_loader):
        # Inputs to Variable type
        input_var = list()
        for j in range(len(input)):
            if torch.cuda.is_available():
                input_var.append(torch.autograd.Variable(input[j]).cuda())
            else:
                input_var.append(torch.autograd.Variable(input[j]))

        
        target_var = list()
        for j in range(len(target)):
            target[j] = torch.tensor(np.array(target[j], dtype=np.uint8))

            if torch.cuda.is_available():
                target[j] = target[j].cuda(non_blocking=True)

            target_var.append(torch.autograd.Variable(target[j]))
        

        # Predictions
        # with torch.no_grad():
        # Output of the model
        if args_dict.append == 'append':
            output = model((input_var[0], target[1]))
        else:
            output = model(input_var[0])
        if symbol_task:
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
        acc_symbols = symbols_detected / symbols_possible
        acc_absence = absence_detected / absence_possible

        print('Symbols detected {acc}'.format(acc=acc_symbols))
        print('Absence detected {acc}'.format(acc=acc_absence))

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
    mtl_mode = args_dict.att == 'all'
    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    # Load classes
    
    
    if args_dict.att == 'type':
        att2i = type2idx
        num_classes = len(att2i)
    elif args_dict.att == 'school':
        att2i = school2idx
        num_classes = len(att2i)
    elif args_dict.att == 'time':
        att2i = time2idx
        num_classes = len(att2i)
    elif args_dict.att == 'author':
        att2i = author2idx
        num_classes = len(att2i)
    elif args_dict.att == 'all':
        att2i = [type2idx, school2idx, time2idx, author2idx]
        num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]

    if args_dict.embedds == 'clip':
        N_CLUSTERS = 77
    else:
        N_CLUSTERS = args_dict.clusters

    # Define model
    if args_dict.embedds == 'graph':
        if args_dict.append != 'append':
            model = KGM(len(att2i), end_dim=N_CLUSTERS, model=args_dict.architecture)
        else:
            model = KGM_append(len(att2i), end_dim=N_CLUSTERS, model=args_dict.architecture)
    else:
        if args_dict.append != 'append':
            model = KGM(num_classes, end_dim=N_CLUSTERS, model=args_dict.architecture, multi_task=args_dict.att=='all')
        else:
            model = KGM_append(len(att2i), end_dim=N_CLUSTERS, model=args_dict.architecture)

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
    if mtl_mode:
        semart_train_loader = ArtDatasetMTL(args_dict, set='train', att2i=att2i, transform=train_transforms, clusters=N_CLUSTERS, k=k)
        semart_val_loader = ArtDatasetMTL(args_dict, set='val', att2i=att2i, transform=val_transforms, clusters=N_CLUSTERS, k=k)
    else:
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
        trainEpoch(args_dict, train_loader, model, loss, optimizer, epoch, symbol_task=args_dict.symbol_task, final_epoch=False)

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
    try:
        epoch
    except:
        epoch = args_dict.nepochs
    
    trainEpoch(args_dict, train_loader, model, loss, optimizer, epoch, symbol_task=args_dict.symbol_task, final_epoch=True)


def train_multitask_classifier(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]

    # Define model
    print(num_classes)
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
            transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(256),                         # we get only the center of that rescaled
            transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
            transforms.RandomHorizontalFlip(),                  # random horizontal flip (data augmentation)
            transforms.ToTensor(),                              # to pytorch tensor
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

    if args_dict.model == 'mtl':
        train_multitask_classifier(args_dict)
    elif args_dict.model == 'kgm':
        train_knowledgegraph_classifier(args_dict)
    else:
        assert False, 'Incorrect model type'

