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
import lenet



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
    mtl_mode = args_dict.att == 'all'
    # switch to train mode
    model.train()
    actual_index = 0
    grad_classifier_path = args_dict.grad_cam_model_path
    checkpoint = torch.load(grad_classifier_path)
    
    lenet_model = lenet.LeNet([args_dict.gradcam_size, args_dict.gradcam_size, 3], [4, 2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        lenet_model.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        lenet_model.load_state_dict(checkpoint)

    lenet_model = lenet_model.to(device)
    lenet_model.eval()

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
    

