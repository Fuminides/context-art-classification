import warnings
import os
import utils
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix

from dataloader_kgm import ArtDatasetKGM
from model_gcn import NODE2VEC_OUTPUT

from params import get_parser
from train import run_train
from semart_test import run_test


def gen_embeds(args_dict):
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
    
    from PIL import Image
    from model_gcn import VisEncoder
    node2vec_emb = pd.read_csv('Data/semart.emd', skiprows=1, sep=' ', header=None, index_col=0)
    semart_edge_list = pd.read_csv('Data/kg_semart.csv', index_col=None, sep=' ')
    semart_categories_keys = pd.read_csv('Data/kg_keys.csv', index_col=None, sep=' ')
    dict_keys = {x: y for _, (y, x) in semart_categories_keys.iterrows()}
    train_df = pd.read_csv(args_dict.dir_dataset + r'/semart_train.csv', sep='\t', encoding='latin1')
    n_samples = semart_edge_list.max().max()
    
    vis_encoder = VisEncoder()
    vis_encoder.load_weights()
    
    feature_matrix = np.zeros(node2vec_emb)
    print('Starting the process... ')
    i=0
    for sample_ix in node2vec_emb.index:
        i += 1
        try:
            dict_keys[sample_ix]
            feature_matrix[sample_ix, :] = node2vec_emb.loc[sample_ix]
        except KeyError:
            image_path = train_df.loc[sample_ix]['IMAGE FILE']
            image = Image.open(image_path).convert('RGB')
            image = transforms(image)
            feature_matrix[sample_ix, :] = vis_encoder.reduce(image)
        
        if i % 1000 == 0:
            print('Sample ' + str(i) + 'th out of ' + str(len(node2vec_emb.index)))
    
    return feature_matrix
    
def vis_encoder_gen(args_dict):
    '''
    Trains the model (autoencoder) to compute the reduced visual emebeddings.

    Args:
        args_dict (TYPE): DESCRIPTION.

    Returns:
        None.

    '''    
    def save_model(args_dict, state):
        directory = args_dict.dir_model + "Reduce/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + 'reduce_' + str(NODE2VEC_OUTPUT) + '_best_model.pth.tar'
        torch.save(state, filename)
    
    import torch
    import torch.nn as nn
    import model_gcn as mgcn
    from torchvision import transforms
    from attributes import load_att_class

    epochs = args_dict.nepochs

    model = mgcn.VisEncoder()
    # Load the model
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load transforms and data loader
    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    train_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.RandomHorizontalFlip(),                  # random horizontal flip (data augmentation)
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])
    
    semart_train_loader = ArtDatasetKGM(args_dict, att_name='type', set='train', att2i=type2idx, transform=train_transforms)
    semart_val_loader = ArtDatasetKGM(args_dict, att_name='type', set='val', att2i=type2idx,
                                        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        semart_train_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    val_loader = torch.utils.data.DataLoader(
        semart_val_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
    # train loop
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                             weight_decay=1e-5)

    best_val = np.Inf
    pat_track = 0
    
    for epoch in range(epochs):

        print('Epoch: ' + str(epoch) + '/' + str(epochs))

        ### TRAIN
        for batch_idx, (input, target) in enumerate(train_loader):

            # Inputs to Variable type
            input_var = list()
            for j in range(len(input)):
                if torch.cuda.is_available():
                    input_var.append(torch.autograd.Variable(input[j]).cuda())
                else:
                    input_var.append(torch.autograd.Variable(input[j]))

            target = model.gen_target(input_var[0])
            output = model(input_var[0])
            loss = criterion(output, target)

            # Backpropagate loss and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        # EVAL
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
                target = model.gen_target(input_var[0])
                output = model(input_var[0])
                perfval = criterion(output, target)

            # check patience
            if perfval <= best_val:
                pat_track += 1
            else:
                pat_track = 0
            if pat_track >= args_dict.patience:
                break
    
            # save if it is the best model
            is_best = perfval > best_val
            best_val = min(perfval, best_val)
            if is_best:
                save_model(args_dict, {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val': best_val,
                    'optimizer': optimizer.state_dict(),
                    'valtrack': pat_track,
                    'curr_val': best_val,
                })
            print('** Validation: %f (best acc) - %f (current acc) - %d (patience)' % (best_val, perfval, pat_track))

    


if __name__ == "__main__":

    # Load parameters
    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    assert args_dict.att in ['type', 'school', 'time', 'author'], \
        'Incorrect classifier. Please select type, school, time, or author.'

    if args_dict.model == 'mtl':
        args_dict.att = 'all'
    args_dict.name = '{}-{}'.format(args_dict.model, args_dict.att)

    opts = vars(args_dict)
    print('------------ Options -------------')
    for k, v in sorted(opts.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-----------------------------------')

    # Check mode and model are correct
    assert args_dict.mode in ['train', 'test', 'reduce', 'gen_graph_dataset'], 'Incorrect mode. Please select either train or test.'
    assert args_dict.model in ['mtl', 'kgm', 'gcn', 'fcm'], 'Incorrect model. Please select either mlt or kgm.'

    # Run process
    if args_dict.mode == 'train':
        run_train(args_dict)
    elif args_dict.mode == 'test':
        run_test(args_dict)
    elif args_dict.mode == 'reduce':
        vis_encoder_gen(args_dict)
    elif args_dict.mode == 'gen_graph_dataset':
        feature_matrix = gen_embeds(args_dict)
