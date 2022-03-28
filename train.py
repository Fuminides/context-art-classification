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
from model_mtl import MTL
from model_kgm import KGM
from dataloader_mtl import ArtDatasetMTL
from dataloader_kgm import ArtDatasetKGM
from attributes import load_att_class
#from torch_geometric.loader import DataLoader


def print_classes(type2idx, school2idx, timeframe2idx, author2idx):
    print('Att type\t %d classes' % len(type2idx))
    print('Att school\t %d classes' % len(school2idx))
    print('Att time\t %d classes' % len(timeframe2idx))
    print('Att author\t %d classes' % len(author2idx))


def save_model(args_dict, state, type='school', train_feature='kgm'):
    directory = args_dict.dir_model + "%s/"%(args_dict.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + train_feature + '_' + type + '_best_model.pth.tar'
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


def trainEpoch(args_dict, train_loader, model, criterion, optimizer, epoch, extra_params=None):

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

        # Targets to Variable type
        target_var = list()
        for j in range(len(target)):
            if torch.cuda.is_available():
                target[j] = target[j].cuda(non_blocking=True)

            target_var.append(torch.autograd.Variable(target[j]))

        # Output of the model
        if extra_params is None:
            output = model(input_var[0])
        else:
            output = model(input_var[0], extra_params)

        if args_dict.model == 'mtl':
            train_loss = 0.25 * criterion(output[0], target_var[0]) + \
                         0.25 * criterion(output[1], target_var[1]) + \
                         0.25 * criterion(output[2], target_var[2]) + \
                         0.25 * criterion(output[3], target_var[3])
            losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        elif args_dict.model == 'kgm':
            class_loss = criterion[0](output[0], target_var[0])
            encoder_loss = criterion[1](output[1], target_var[1])
            train_loss = args_dict.lambda_c * class_loss + \
                         args_dict.lambda_e * encoder_loss
            losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Print info
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))

    # Plot
    #plotter.plot('closs', 'train', 'Class Loss', epoch, losses.avg)


def valEpoch(args_dict, val_loader, model, criterion, epoch):

    # object to store & plot the losses
    losses = utils.AverageMeter()

    # switch to evaluation mode
    model.eval()
    for batch_idx, (input, target) in enumerate(val_loader):

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
            if torch.cuda.is_available():
                target[j] = target[j].cuda(non_blocking=True)

            target_var.append(torch.autograd.Variable(target[j]))

        # Predictions
        with torch.no_grad():
            output = model(input_var[0])

        if args_dict.model == 'mtl':
            _, pred_type = torch.max(output[0], 1)
            _, pred_school = torch.max(output[1], 1)
            _, pred_time = torch.max(output[2], 1)
            _, pred_author = torch.max(output[3], 1)

            val_loss = 0.25 * criterion(output[0], target_var[0]) + \
                         0.25 * criterion(output[1], target_var[1]) + \
                         0.25 * criterion(output[2], target_var[2]) + \
                         0.25 * criterion(output[3], target_var[3])
            losses.update(val_loss.data.cpu().numpy(), input[0].size(0))

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

        elif args_dict.model == 'kgm':
            _, predicted = torch.max(output[0], 1)
            train_loss = criterion[0](output[0], target_var[0])
            losses.update(train_loss.data.cpu().numpy(), input[0].size(0))

            # Save predictions to compute accuracy
            if batch_idx==0:
                out = predicted.data.cpu().numpy()
                label = target[0].cpu().numpy()
            else:
                out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
                label = np.concatenate((label,target[0].cpu().numpy()),axis=0)

    # Accuracy
    if args_dict.model == 'mtl':
        acc_type = np.sum(out_type == label_type)/len(out_type)
        acc_school = np.sum(out_school == label_school) / len(out_school)
        acc_tf = np.sum(out_time == label_tf) / len(out_time)
        acc_author = np.sum(out_author == label_author) / len(out_author)
        acc = np.mean((acc_type, acc_school, acc_tf, acc_author))
    elif args_dict.model == 'kgm':
        acc = np.sum(out == label) / len(out)

    # Print validation info
    print('Validation set: Average loss: {:.4f}\t'
          'Accuracy {acc}'.format(losses.avg, acc=acc))
    #plotter.plot('closs', 'val', 'Class Loss', epoch, losses.avg)
    #plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)

    # Return acc
    return acc


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
        model = KGM(len(att2i))
    else:
        model = KGM(len(att2i), end_dim=N_CLUSTERS)

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
    semart_train_loader = ArtDatasetKGM(args_dict, set='train', att2i=att2i, att_name=args_dict.att, transform=train_transforms, embedds=args_dict.embedds, clusters=N_CLUSTERS, k=k)
    semart_val_loader = ArtDatasetKGM(args_dict, set='val', att2i=att2i, att_name=args_dict.att, transform=val_transforms, embedds=args_dict.embedds, clusters=N_CLUSTERS, k=k)

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
            }, type=args_dict.att, train_feature=args_dict.embedds)
        print('** Validation: %f (best acc) - %f (current acc) - %d (patience)' % (best_val, accval, pat_track))


def train_multitask_classifier(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]

    # Define model
    model = MTL(num_classes)
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
            }, type=args_dict.att, train_feature=args_dict.embedds)

        print('** Validation: %f (best acc) - %f (current acc) - %d (patience)' % (best_val, accval, pat_track))


def _load_labels(df_path, att2i):
    def class_from_name(name, vocab):

        if name in vocab:
            idclass= vocab[name]
        else:
            idclass = vocab['UNK']

        return idclass

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
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
    from model_gcn import GCN, NODE2VEC_OUTPUT

    
    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]
    
    # Load semart knowledge graphs
    train_edge_list = pd.read_csv(args_dict.edge_list_train, index_col=None, sep=' ')
    val_edge_list = pd.read_csv(args_dict.edge_list_val, index_col=None, sep=' ')
    test_edge_list = pd.read_csv(args_dict.edge_list_test, index_col=None, sep=' ')
    
    # Use the kgs to generate a sparse matrix
    total_edge_list = pd.concat([train_edge_list, val_edge_list, test_edge_list], axis=0)
    n_samples = int(total_edge_list.max().max()+1)
    '''adj_sparse = dok_matrix((n_samples, n_samples), dtype=np.int8)
    for row in range(train_edge_list.shape[0]):
        emisor = total_edge_list.iloc[row, 0]
        receptor = total_edge_list.iloc[row, 1]
        adj_sparse[emisor, receptor] = 1'''

    tensor_total_edge_list = torch.tensor(np.array(total_edge_list), dtype=torch.long)
    

    # Load the feature matrix from the vis+node2vec representations
    train_feature_matrix = pd.read_csv(args_dict.feature_matrix, sep=' ')
    val_feature_matrix = pd.read_csv(args_dict.val_feature_matrix, sep=' ')
    test_feature_matrix = pd.read_csv(args_dict.test_feature_matrix, sep=' ')
    
    total_samples = torch.tensor(np.array(pd.concat([train_feature_matrix, val_feature_matrix, test_feature_matrix], axis=0)))
    n_samples = total_samples.shape[0]

    # Gen the train/val/test indexes
    train_mask = np.array([0] * n_samples)
    train_mask[0:train_feature_matrix.shape[0]] = 1
    train_mask = torch.tensor(train_mask, dtype=torch.uint8)
    
    val_mask = np.array([0] * n_samples)
    val_mask[train_feature_matrix.shape[0]:train_feature_matrix.shape[0]+val_feature_matrix.shape[0]] = 1
    val_mask = torch.tensor(val_mask, dtype=torch.uint8)
    
    test_mask = np.array([0] * n_samples)
    test_mask[-test_feature_matrix.shape[0]:] = 1
    test_mask = torch.tensor(test_mask, dtype=torch.uint8)

    #Load labels
    
    #Load all the data as Data object for pytorch geometric
    data = Data(x=total_samples, edge_index=tensor_total_edge_list)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Define model
    model = GCN(NODE2VEC_OUTPUT, int(NODE2VEC_OUTPUT / 2), int(NODE2VEC_OUTPUT / 4), num_classes)
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

    # Resume training if needed
    best_val, model, optimizer = resume(args_dict, model, optimizer)
    # Loss and optimizer
    if torch.cuda.is_available():
        class_loss = nn.CrossEntropyLoss().cuda()
    else:
        class_loss = nn.CrossEntropyLoss()

    # Dataloaders for training and validation
    target_var_train = _load_labels(args_dict.dir_dataset + '/semart_train.csv', att2i)
    target_var_val = _load_labels(args_dict.dir_dataset + '/semart_val.csv', att2i)
    target_var_test = _load_labels(args_dict.dir_dataset + '/semart_test.csv', att2i)


    # Now, let's start the training process!
    print_classes(type2idx, school2idx, time2idx, author2idx)
    losses = utils.AverageMeter()
    print('Start training GCN model...')
    pat_track = 0

    for epoch in range(args_dict.start_epoch, args_dict.nepochs):
        print(epoch)
        # Compute a training epoch
        optimizer.zero_grad()
        output = model(data.x[data.train_mask], data.edge_index)
        print('Output computed')
        train_loss = 0.25 * criterion(output[0], target_var_train[0]) + \
                     0.25 * criterion(output[1], target_var_train[1]) + \
                     0.25 * criterion(output[2], target_var_train[2]) + \
                     0.25 * criterion(output[3], target_var_train[3])
        print('Loss computed')
        losses.update(train_loss.data.cpu().numpy(), input[0].size(0))
        train_loss.backward()
        optimizer.step()

        # Compute a validation epoch
        # accval = valEpoch(args_dict, val_loader, model, class_loss, epoch)
        output = model(data.x[data.val_mask])
        _, pred_type = torch.max(output[0], 1)
        _, pred_school = torch.max(output[1], 1)
        _, pred_time = torch.max(output[2], 1)
        _, pred_author = torch.max(output[3], 1)

        # Save predictions to compute accuracy
   
        out_type = pred_type.data.cpu().numpy()
        out_school = pred_school.data.cpu().numpy()
        out_time = pred_time.data.cpu().numpy()
        out_author = pred_author.data.cpu().numpy()
        label_type = target_var_val[0].cpu().numpy()
        label_school = target_var_val[1].cpu().numpy()
        label_tf = target_var_val[2].cpu().numpy()
        label_author = target_var_val[3].cpu().numpy()
   
        acc_type = np.sum(out_type == label_type)/len(out_type)
        acc_school = np.sum(out_school == label_school) / len(out_school)
        acc_tf = np.sum(out_time == label_tf) / len(out_time)
        acc_author = np.sum(out_author == label_author) / len(out_author)
        accval = np.mean((acc_type, acc_school, acc_tf, acc_author))
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
            }, type=args_dict.att, train_feature=args_dict.embedds)

        print('** Validation: %f (best acc) - %f (current acc) - %d (patience)' % (best_val, accval, pat_track))


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
    elif args_dict.model == 'gcn':
        train_gcn_classifier(args_dict)
    else:
        assert False, 'Incorrect model type'

