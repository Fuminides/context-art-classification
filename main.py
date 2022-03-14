import warnings
import utils
warnings.filterwarnings("ignore")

from dataloader_kgm import ArtDatasetKGM

from params import get_parser
from train import run_train
from semart_test import run_test

def vis_encoder_gen(args_dict):
    import torch
    import torch.nn as nn
    import model_gcn as mgcn
    from torchvision import transforms
    from attributes import load_att_class

    epochs = args_dict.epochs

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
    i=0

    for epoch in epochs:

        print('Epoch: ' + str(epoch) + '/' + epochs)

        ### TRAIN
        for batch_idx, (input, target) in enumerate(train_loader):

            # Inputs to Variable type
            input_var = list()
            for j in range(len(input)):
                if torch.cuda.is_available():
                    input_var.append(torch.autograd.Variable(input[j]).cuda())
                else:
                    input_var.append(torch.autograd.Variable(input[j]))

            output = model(input_var[0])
            target = model.gen_target(input_var[0])
            loss = criterion(output, target)

            # Backpropagate loss and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1


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

        # Once train is finished we generate the embeddings for all the images
    


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
    assert args_dict.mode in ['train', 'test', 'reduce'], 'Incorrect mode. Please select either train or test.'
    assert args_dict.model in ['mtl', 'kgm', 'gcn', 'fcm'], 'Incorrect model. Please select either mlt or kgm.'

    # Run process
    if args_dict.mode == 'train':
        run_train(args_dict)
    elif args_dict.mode == 'test':
        run_test(args_dict)
    elif args_dict.mode == 'reduce':
        vis_encoder_gen(args_dict)
