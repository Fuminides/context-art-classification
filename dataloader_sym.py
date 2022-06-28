from unicodedata import name
import numpy as np
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

        myth_edges = an.load_edges_myth()
        myth_entities = np.unique(list(myth_edges['Source']) + list(myth_edges['Target']))
        args_dict.canon_list = myth_entities
        self.symbol_context, self.paintings_names, self.symbols_names = an.load_semart_symbols(args_dict)
        print('Symbol mat: ' + str(self.symbol_context.shape))
        self.semart_Gallery = an.Gallery(self.symbols_names, self.paintings_names, self.symbol_context, args_dict.dir_dataset)



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
        imagepath = self.imagefolder +  '/' + self.imageurls[index]
        image = Image.open(imagepath).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Attribute class
        try:
            symbols = self.symbol_context[index, :]
        except Exception as e:
            print(e, self.set, index)

        return [image], symbols

#def filter_symbols():
if __name__ == '__main__':
    import params
    

    from attributes import load_att_class
    args_dict = params.get_parser()

    args_dict.dir_data = 'Data'
    args_dict.mode = 'train'
    args_dict.vocab_type = 'type2ind.csv'
    args_dict.vocab_school = 'school2ind.csv'
    args_dict.vocab_time = 'time2ind.csv'
    args_dict.vocab_author = 'author2ind.csv'
    args_dict.embedds = 'tfidf'
    args_dict.dir_dataset = '/home/javierfumanal/Documents/GitHub/SemArt/'
    args_dict.csvtrain = 'semart_train.csv'
    args_dict.csvval = 'semart_val.csv'
    args_dict.dir_images = 'Images'


    type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)

    semart_val_loader = ArtDatasetSym(args_dict, set=args_dict.mode)
    for batch_idx, (input, target) in enumerate(semart_val_loader):
        print(batch_idx, len(input), target.shape)