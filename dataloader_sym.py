import torch.utils.data as data
import pandas as pd
import os

from PIL import Image

import annotation_analysis as an


class ArtDatasetSym(data.Dataset):

    def __init__(self, args_dict, set, transform = None):
        """
        Args:
            args_dict: parameters dictionary
            set: 'train', 'val', 'test'
            transform: data transform
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

        self.imagefolder = os.path.join(args_dict.dir_dataset, args_dict.dir_images)
        self.transform = transform

        self.imageurls = list(df['IMAGE_FILE'])

        self.symbol_context, self.paintings_names, self.symbols_names = an.__load_semart_proxy(mode='train')
        self.semart_Gallery = an.Gallery(self.symbols_names, self.paintings_names, self.symbol_context, an.args_dict.dir_dataset)


    def __len__(self):
        return len(self.imageurls)


    def class_from_name(self, vocab, name):

        if name in vocab:
            idclass= vocab[name]
        else:
            idclass = vocab['UNK']

        return idclass


    def __getitem__(self, index):

        # Load image & apply transformation
        imagepath = self.imagefolder + self.imageurls[index]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Attribute class
        symbols = self.symbol_context[index, :]

        return [image], symbols
