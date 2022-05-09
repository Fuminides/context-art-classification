from __future__ import division

import numpy as np
import torch
from torchvision import transforms

from model_mtl import MTL
from model_kgm import KGM
from dataloader_mtl import ArtDatasetMTL
from dataloader_kgm import ArtDatasetKGM
from attributes import load_att_class

from model_gcn import GCN, NODE2VEC_OUTPUT

def test_knowledgegraph(args_dict):

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

    # Define model
    if args_dict.embedds == 'graph':
        model = KGM(len(att2i))
    else:
        model = KGM(len(att2i), end_dim=args_dict.clusters)

    if torch.cuda.is_available():#args_dict.use_gpu:
        model.cuda()

    # Load best model
    try:
        print("=> loading checkpoint '{}'".format(args_dict.model_path))
        checkpoint = torch.load(args_dict.model_path)
        args_dict.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args_dict.model_path, checkpoint['epoch']))
    except RuntimeError:
        print('No checkpoint available')
        args_dict.start_epoch = 0


    # Data transformation for test
    test_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for test
    if torch.cuda.is_available():
        test_loader = torch.utils.data.DataLoader(
            ArtDatasetKGM(args_dict, set='test', att2i=att2i, att_name=args_dict.att, transform=test_transforms),
            batch_size=args_dict.batch_size, shuffle=False, pin_memory=(not args_dict.no_cuda), num_workers=args_dict.workers)
    else:
        test_loader = torch.utils.data.DataLoader(
            ArtDatasetKGM(args_dict, set='test', att2i=att2i, att_name=args_dict.att, transform=test_transforms),
            batch_size=args_dict.batch_size, shuffle=False, pin_memory=False,
            num_workers=args_dict.workers)

    # Switch to evaluation mode & compute test samples embeddings
    model.eval()
    for i, (input, target) in enumerate(test_loader):

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
        with torch.no_grad():
            # Output of the model
            if args_dict.append == 'append':
                output = model(input_var[0], target[1])
            else:
                output = model(input_var[0])

            #outsoftmax = torch.nn.functional.softmax(output[0])
        conf, predicted = torch.max(output, 1)

        # Store embeddings
        if i==0:
            out = predicted.data.cpu().numpy()
            label = target[0].cpu().numpy()
            scores = conf.data.cpu().numpy()
        else:
            out = np.concatenate((out,predicted.data.cpu().numpy()), axis=0)
            label = np.concatenate((label,target[0].cpu().numpy()), axis=0)
            scores = np.concatenate((scores, conf.data.cpu().numpy()), axis=0)

    # Compute Accuracy
    acc = np.sum(out == label)/len(out)
    print('Model %s\tTest Accuracy %.03f' % (args_dict.model_path, acc))


def test_multitask(args_dict):

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]

    model = MTL(num_classes)
    if torch.cuda.is_available():
        model.cuda()

    # Load best model
    print("=> loading checkpoint '{}'".format(args_dict.model_path))
    checkpoint = torch.load(args_dict.model_path)
    args_dict.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args_dict.model_path, checkpoint['epoch']))

    # Data transformation for test
    test_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for test
    if torch.cuda.is_available():
        test_loader = torch.utils.data.DataLoader(
            ArtDatasetMTL(args_dict, set='test', att2i=att2i, transform=test_transforms),
            batch_size=args_dict.batch_size, shuffle=False, pin_memory=(not args_dict.no_cuda), num_workers=args_dict.workers)
    else:
        test_loader = torch.utils.data.DataLoader(
            ArtDatasetMTL(args_dict, set='test', att2i=att2i, transform=test_transforms),
            batch_size=args_dict.batch_size, shuffle=False, pin_memory=False,
            num_workers=args_dict.workers)

    # Switch to evaluation mode & compute test
    model.eval()
    for i, (input, target) in enumerate(test_loader):

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
        with torch.no_grad():
            output = model(input_var[0])
        _, pred_type = torch.max(output[0], 1)
        _, pred_school = torch.max(output[1], 1)
        _, pred_time = torch.max(output[2], 1)
        _, pred_author = torch.max(output[3], 1)

        # Store outputs
        if i==0:
            out_type = pred_type.data.cpu().numpy()
            out_school = pred_school.data.cpu().numpy()
            out_time = pred_time.data.cpu().numpy()
            out_author = pred_author.data.cpu().numpy()
            label_type = target[0].cpu().numpy()
            label_school = target[1].cpu().numpy()
            label_time = target[2].cpu().numpy()
            label_author = target[3].cpu().numpy()
        else:
            out_type = np.concatenate((out_type,pred_type.data.cpu().numpy()),axis=0)
            out_school = np.concatenate((out_school, pred_school.data.cpu().numpy()), axis=0)
            out_time = np.concatenate((out_time, pred_time.data.cpu().numpy()), axis=0)
            out_author = np.concatenate((out_author, pred_author.data.cpu().numpy()), axis=0)
            label_type = np.concatenate((label_type,target[0].cpu().numpy()),axis=0)
            label_school = np.concatenate((label_school,target[1].cpu().numpy()),axis=0)
            label_time = np.concatenate((label_time,target[2].cpu().numpy()),axis=0)
            label_author = np.concatenate((label_author,target[3].cpu().numpy()),axis=0)

    # Compute Accuracy
    acc_type = np.sum(out_type == label_type)/len(out_type)
    acc_school = np.sum(out_school == label_school) / len(out_school)
    acc_tf = np.sum(out_time == label_time) / len(out_time)
    acc_author = np.sum(out_author == label_author) / len(out_author)

    # Print test accuracy
    print('------------ Test Accuracy -------------')
    print('Type Accuracy %.03f' % acc_type)
    print('School Accuracy %.03f' % acc_school)
    print('Timeframe Accuracy %.03f' % acc_tf)
    print('Author Accuracy %.03f' % acc_author)
    print('----------------------------------------')
  
def test_gcn(args_dict):
    from train import _load_labels
    from torch_geometric.data import Data
    import pandas as pd

    # Load classes
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    num_classes = [len(type2idx), len(school2idx), len(time2idx), len(author2idx)]
    att2i = [type2idx, school2idx, time2idx, author2idx]

    

    # Data transformation for test
    test_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(224),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

    # Data Loaders for test
    test_edge_list = pd.read_csv(args_dict.edge_list_test, index_col=None, sep=' ', header=None)
    
    # Load semart knowledge graphs
    train_edge_list = pd.read_csv(args_dict.edge_list_train, index_col=None, sep=' ', header=None)
    val_edge_list = pd.read_csv(args_dict.edge_list_val, index_col=None, sep=' ', header=None)
    test_edge_list = pd.read_csv(args_dict.edge_list_test, index_col=None, sep=' ', header=None)

    #test_edge_list = pd.concat([train_edge_list, test_edge_list], axis=0)
    tensor_test_edge_list = torch.tensor(np.array(test_edge_list).reshape((2, test_edge_list.shape[0])), dtype=torch.long)
    total_edge_list = pd.concat([train_edge_list, val_edge_list, test_edge_list], axis=0)
    tensor_total_edge_list = torch.tensor(np.array(total_edge_list).reshape((2, total_edge_list.shape[0])), dtype=torch.long)

    #  Load the feature matrix from the vis+node2vec representations
    train_feature_matrix = pd.read_csv(args_dict.feature_matrix, sep=' ', header=None, skiprows=1, index_col=0)
    val_feature_matrix = pd.read_csv(args_dict.val_feature_matrix, sep=' ',  header=None, skiprows=1, index_col=0)
    test_feature_matrix = pd.read_csv(args_dict.test_feature_matrix, sep=' ',  header=None, skiprows=1, index_col=0)

    total_samples = torch.tensor(np.array(pd.concat([train_feature_matrix, val_feature_matrix, test_feature_matrix], axis=0))).float()
    n_samples = total_samples.shape[0]

    target_var_test = _load_labels(args_dict.dir_dataset + '/semart_test.csv', att2i)

    # Gen the train/val/test indexes
    train_mask = np.array([0] * n_samples)
    train_mask[0:train_feature_matrix.shape[0]] = 1
    train_mask = torch.tensor(train_mask, dtype=torch.uint8)
    
    val_mask = np.array([0] * n_samples)
    val_mask[train_feature_matrix.shape[0]:train_feature_matrix.shape[0]+val_feature_matrix.shape[0]] = 1
    val_mask = torch.tensor(val_mask, dtype=torch.uint8)
    
    test_mask = np.array([0] * n_samples)
    test_mask[-len(target_var_test[0]):] = 1
    test_mask = torch.tensor(test_mask, dtype=torch.uint8)

    if torch.cuda.is_available():
        total_samples = total_samples.cuda()
        train_edge_list = torch.tensor(np.array(train_edge_list).reshape(2, train_edge_list.shape[0])).cuda()
        tensor_total_edge_list = tensor_total_edge_list.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    #Load all the data as Data object for pytorch geometric
    data = Data(x=total_samples, edge_index=tensor_total_edge_list)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Define model
    model = GCN(NODE2VEC_OUTPUT, 16, num_classes)
    if torch.cuda.is_available():
        model.cuda()
    
    # Load best model
    print("=> loading checkpoint '{}'".format(args_dict.model_path))
    checkpoint = torch.load(args_dict.model_path)
    args_dict.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args_dict.model_path, checkpoint['epoch']))


    # Dataloaders for training and validation

    # Switch to evaluation mode & compute test
    model.eval()
    output = model(data.x, data.edge_index)
    pred_type = torch.argmax(output[0][data.test_mask], 1)
    pred_school = torch.argmax(output[1][data.test_mask], 1)
    pred_time = torch.argmax(output[2][data.test_mask], 1)
    pred_author = torch.argmax(output[3][data.test_mask], 1)
    
    # Save predictions to compute accuracy
    out_type = pred_type.data.cpu().numpy()
    out_school = pred_school.data.cpu().numpy()
    out_time = pred_time.data.cpu().numpy()
    out_author = pred_author.data.cpu().numpy()
    label_type = target_var_test[0]#.cpu().numpy()
    label_school = target_var_test[1]#.cpu().numpy()
    label_tf = target_var_test[2]#.cpu().numpy()
    label_author = target_var_test[3]#.cpu().numpy()

    acc_type = np.mean(np.equal(out_type, label_type))
    acc_school = np.mean(np.equal(out_school, label_school))
    acc_tf = np.mean(np.equal(out_time, label_tf))
    acc_author = np.mean(np.equal(out_author, label_author)) 
    accval = np.mean((acc_type, acc_school, acc_tf, acc_author))

    # Print test accuracy
    print('------------ Test Accuracy -------------')
    print('Type Accuracy %.03f' % acc_type)
    print('School Accuracy %.03f' % acc_school)
    print('Timeframe Accuracy %.03f' % acc_tf)
    print('Author Accuracy %.03f' % acc_author)
    print('Average accuracy %.03f' % accval)
    print('----------------------------------------')



def run_test(args_dict):

    if args_dict.model == 'mtl':
        test_multitask(args_dict)
    if args_dict.model == 'gcn':
        test_gcn(args_dict)
    elif args_dict.model == 'kgm':
        test_knowledgegraph(args_dict)
    else:
        assert False, 'Incorrect model type'

