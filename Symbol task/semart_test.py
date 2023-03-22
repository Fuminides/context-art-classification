from __future__ import division

import numpy as np
import torch
from torchvision import transforms
from dataloader_sym import ArtDatasetSym

from attributes import load_att_class
from model_sym import SymModel

import pandas as pd
#from model_gcn import GCN, 
NODE2VEC_OUTPUT = 128


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
        [quantity, size] = lenet_model(torch.unsqueeze(grad_cam_image, 0))

        res_quant[ix] = quantity.detach().cpu().numpy()
        res_size[ix] = size.detach().cpu().numpy()

    pd.DataFrame(res_quant).to_csv('./DeepFeatures/sym_grad_cam_test_quant_' + str(batch_idx) + '_' + str(args_dict.att) + '_' + str(args_dict.embedds) + '.csv')
    pd.DataFrame(res_size).to_csv('./DeepFeatures/sym_grad_cam_test_size_' + str(batch_idx) + '_' + str(args_dict.att) + '_' + str(args_dict.embedds) + '.csv')
    

def test_knowledgegraph(args_dict):

    # Define model
    semart_train_loader = ArtDatasetSym(args_dict, set='train', transform=None)
    model = SymModel(len(semart_train_loader.symbols_names), model=args_dict.architecture)

    if torch.cuda.is_available():#args_dict.use_gpu:
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
    semart_train_loader = ArtDatasetSym(args_dict, set='train', transform=None)
    test_loader = torch.utils.data.DataLoader(
                ArtDatasetSym(args_dict, set='test', transform=test_transforms, canon_list=semart_train_loader.symbols_names),
                batch_size=args_dict.batch_size, shuffle=False, pin_memory=(not args_dict.no_cuda), num_workers=args_dict.workers)


    # Switch to evaluation mode & compute test samples embeddings
    model.eval()

    acc_sample = 0
    symbols_detected = 0
    symbols_possible = 0
    acc_possible = 0
    absence_detected = 0
    absence_possible = 0


    grad_classifier_path = args_dict.grad_cam_model_path
    checkpoint_lenet = torch.load(grad_classifier_path)
    
    # Lenet is for gradcam classification
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
            target_var = torch.tensor(np.array(target, dtype=np.float), dtype=torch.float32)

            if j == 0:
                target[j] = torch.tensor(np.array(target[j].cpu(), dtype=np.int32))
            else:
                target[j] = target_var # torch.tensor(target[j])
            
            if torch.cuda.is_available():
                target[j] = target[j].cuda(non_blocking=True)

            target_var.append(torch.autograd.Variable(target[j]))
        
        # Output of the mode
        output = model(input_var[0])

        pred = output > 0.5
        label_actual = target.cpu().numpy()
        symbols_detected += np.sum(np.logical_and(pred.cpu().numpy(), label_actual), axis=None) 
        symbols_possible += np.sum(label_actual, axis=None)
        acc_sample += np.sum(np.equal(pred.cpu().numpy(), label_actual), axis=None)
        acc_possible += pred.shape[0] * pred.shape[1]
        absence_detected += np.sum(np.logical_and(np.logical_not(pred.cpu().numpy()), np.logical_not(label_actual)), axis=None) 
        absence_possible += np.sum(np.logical_not(label_actual), axis=None)

        # extract_grad_cam_features(model, input_var[0], target_var, args_dict, i, lenet_model)


    # Compute Accuracy    
    acc = acc_sample / acc_possible
    acc_symbols = symbols_detected / symbols_possible
    acc_absence = absence_detected / absence_possible

    print('Global accuracy {acc}'.format(acc=acc))
    print('Symbols detected {acc}'.format(acc=acc_symbols))
    print('Absence detected {acc}'.format(acc=acc_absence))

    




def run_test(args_dict):

    test_knowledgegraph(args_dict)

