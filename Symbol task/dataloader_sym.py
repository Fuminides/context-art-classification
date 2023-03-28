import random
import math 
from torchvision import transforms

from unicodedata import name
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import os

from PIL import Image

import annotation_analysis as an


class ArtDatasetSym(data.Dataset):

    def __init__(self, args_dict, set, transform=None, symbol_detect=None, imbalance_ratio=1, fiability_threshold=0.9):
        """
        Args:
            args_dict: parameters dictionary
            set: 'train', 'val', 'test'
            transform: data transform
            symbol_detect: list of symbols to detect (integers)
            imbalance_ratio: ratio of negative samples to positive samples

        """
        self.args_dict = args_dict
        self.set = set


        # Load data
        if self.set == 'train':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtrain)
        elif self.set == 'val':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvval)
        elif self.set == 'test':
            textfile = os.path.join(args_dict.dir_dataset, args_dict.csvtest)
        df = pd.read_csv(textfile, delimiter='\t', encoding='Cp1252')
        fiability_path = os.path.join('Data/', 'fiabilities_' + self.set + '.csv')
        self.fiability =pd.read_csv(fiability_path, index_col=0)

        self.fiability_threshold = fiability_threshold
        boolean_index = self.fiability['Fiability'] > self.fiability_threshold
        boolean_index.index = df.index
        df = df.loc[boolean_index]

        
        self.imagefolder = os.path.join(args_dict.dir_dataset, args_dict.dir_images)
        self.transform = transform

        self.imageurls = list(df['IMAGE_FILE'])


        # Load symbols
        painting_symbol_train = pd.read_csv('Symbol task/Data/' + self.set + '_symbol_labels.csv', index_col=0)
        boolean_index.index = self.imageurls
        painting_symbol_train = painting_symbol_train.loc[boolean_index]
        
        self.symbol_context = painting_symbol_train.values
        self.symbols_names = painting_symbol_train.columns
        self.paintings_names = painting_symbol_train.index        

        self.subset = symbol_detect is not None

        if self.subset:
            self.target_index_list = symbol_detect

            self.semart_Gallery = an.Gallery(self.symbols_names, self.paintings_names, self.symbol_context, args_dict.dir_dataset)
            ratios = [self.symbol_context[:, target_x].mean() for target_x in self.target_index_list]

            print('Original target density in ' + set + ': ' + str(ratios))

            self.sum_positive_paintings = np.sum([self.symbol_context.sum(axis=0)[target_x] for target_x in self.target_index_list])
            self.imbalance_ratio = imbalance_ratio
            self.positive_samples_index = np.logical_or.reduce([self.symbol_context[:, symbol] for symbol in self.target_index_list])
            self.positive_samples = [self.imageurls[ix] for ix, x in enumerate(self.positive_samples_index) if x]
            self.negative_samples = [self.imageurls[ix] for ix, x in enumerate(self.positive_samples_index) if not x]

            self.symbol_context = self.symbol_context[:, self.target_index_list]

            self.generate_negative_samples()


    def generate_negative_samples(self):
        if not self.subset:
            raise NameError('Only generate negative samples for balanced targeted subset')
        
        examples_to_generate = int(self.sum_positive_paintings * self.imbalance_ratio)
        generated_negative_samples = np.random.choice(self.negative_samples, examples_to_generate, replace=False)
        self.balanced_imageurls = self.positive_samples + generated_negative_samples.tolist()
        random.shuffle(self.balanced_imageurls)

        self.positive_symbol_context = self.symbol_context[self.positive_samples_index, :]
        self.negative_symbol_context = self.symbol_context[~self.positive_samples_index, :]
        self.balanced_symbol_context = np.concatenate((self.positive_symbol_context, self.negative_symbol_context), axis=0)
        np.random.permutation(self.balanced_symbol_context)



    def __len__(self):
        if not self.subset:
            return len(self.imageurls)
        else:
            return int(self.sum_positive_paintings * (self.imbalance_ratio+1))


    def class_from_name(self, vocab, name):

        if name in vocab:
            idclass= vocab[name]
        else:
            idclass = vocab['UNK']

        return idclass


    def __getitem__(self, index):

        # Load image & apply transformation
        if not self.subset:
            imagepath = self.imagefolder +  '/' + self.imageurls[index]
        else:
            imagepath = self.imagefolder +  '/' + self.balanced_imageurls[index]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Attribute class
        if self.subset:
            symbols_positive = self.balanced_symbol_context[index, :].astype(int)
            
        else:
            symbols_positive = self.symbol_context[index, :].astype(int)
        
        if len(self.target_index_list) == 1:
            symbols_positive = symbols_positive[:, None]

        symbols_negatives = np.logical_not(symbols_positive)
        symbols = np.concatenate((symbols_positive, symbols_negatives), axis=1)

        painting_fiability = self.fiability.loc[self.imageurls[index]][0]
        return image, symbols_positive, painting_fiability

#def filter_symbols():
if __name__ == '__main__':
    import params
    

    args_dict = params.get_parser()

    args_dict.dir_data = 'Data'
    args_dict.mode = 'train'
    args_dict.vocab_type = 'type2ind.csv'
    args_dict.vocab_school = 'school2ind.csv'
    args_dict.vocab_time = 'time2ind.csv'
    args_dict.vocab_author = 'author2ind.csv'
    args_dict.embedds = 'tfidf'
    args_dict.dir_dataset = r'/home/javierfumanal/Documents/GitHub/SemArt'
    args_dict.csvtrain = 'semart_train.csv'
    args_dict.csvval = 'semart_val.csv'
    args_dict.csvtest = 'semart_test.csv'
    args_dict.dir_images = 'Images'
    args_dict.batch_size = 32
    args_dict.targets = [10]

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

    #type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    semart_train_loader = ArtDatasetSym(args_dict, set='train', symbol_detect=args_dict.targets, transform=train_transforms)
    semart_val_loader = ArtDatasetSym(args_dict, set='val', symbol_detect=args_dict.targets, transform=val_transforms) 
    semart_test_loader = ArtDatasetSym(args_dict, set='test', symbol_detect=args_dict.targets, transform=val_transforms) 

    train_loader = torch.utils.data.DataLoader(
        semart_val_loader,
        batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    print('Training loader with %d samples' % train_loader.dataset.__len__())

    symbols_list = []
    for epoch in range(10):
        for i, (images, symbols) in enumerate(train_loader):
            symbols_list.append(symbols)
        
        train_loader.dataset.generate_negative_samples()
    
    assert math.isclose(np.mean([torch.mean(x) for x in symbols_list]) , 0.5, rel_tol=1e-9)
    print('Test passed')