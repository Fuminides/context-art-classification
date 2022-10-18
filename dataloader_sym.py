from unicodedata import name
import numpy as np
import torch.utils.data as data
import pandas as pd
import os

from PIL import Image

import annotation_analysis as an


class ArtDatasetSym(data.Dataset):

    def __init__(self, args_dict, set, transform=None, canon_list=None, symbol_detect=None):
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

        if canon_list is None:
            import re
            pattern = re.compile('[^a-zA-Z]+')
            myth_edges = an.load_edges_myth()
            source_names = np.unique(list(myth_edges['Source']))
            source_names = [pattern.sub('', name) for name in source_names]
            target_names = np.unique(list(myth_edges['Target']))
            target_names = [pattern.sub('', name) for name in target_names]
            myth_entities = np.unique(source_names + target_names)
            args_dict.canon_list = myth_entities
        else:
            args_dict.canon_list = canon_list

        self.symbol_context, self.paintings_names, self.symbols_names = an.load_semart_symbols(args_dict, self.set, strict_names=self.set!='train', )
        print('Symbol mat: ' + str(self.symbol_context.shape), 'Set: ' + self.set, 'Symbol names: ' + str(len(self.symbols_names)))
        
        

        self.subset = symbol_detect is not None
        self.target_index_list = symbol_detect

        self.semart_Gallery = an.Gallery(self.symbols_names, self.paintings_names, self.symbol_context, args_dict.dir_dataset)
        ratios = [self.semart_Gallery.ratio_symbol(target_x) for target_x in self.target_index_list]
        print('Target density: ' + str(ratios))



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
            if self.subset:
                symbols = self.symbol_context[index, :]
                symbols = np.array([symbols[x] for x in self.target_index_list])
            else:
                symbols = self.symbol_context[index, :]
        except Exception as e:
            print(e, self.set, index)

        return [image], np.squeeze(symbols)

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
    args_dict.dir_dataset = r'C:\Users\jf22881\Documents\SemArt'
    args_dict.csvtrain = 'semart_train.csv'
    args_dict.csvval = 'semart_val.csv'
    args_dict.csvtest = 'semart_test.csv'
    args_dict.dir_images = 'Images'
    args_dict.targets = [10, 20 ,30]

    #type2idx, school2idx, time2idx, author2idx = load_att_class(args_dict)
    semart_train_loader = ArtDatasetSym(args_dict, set='train', symbol_detect=args_dict.targets)
    semart_val_loader = ArtDatasetSym(args_dict, set='val', symbol_detect=args_dict.targets, canon_list=semart_train_loader.symbols_names) 
    semart_test_loader = ArtDatasetSym(args_dict, set='test', symbol_detect=args_dict.targets, canon_list=semart_train_loader.symbols_names) 

    semart_Gallery = an.Gallery(semart_train_loader.symbols_names, semart_train_loader.paintings_names, semart_train_loader.symbol_context, args_dict.dir_dataset)
    val_Gallery = an.Gallery(semart_val_loader.symbols_names, semart_val_loader.paintings_names, semart_val_loader.symbol_context, args_dict.dir_dataset)
    test_Gallery = an.Gallery(semart_test_loader.symbols_names, semart_test_loader.paintings_names, semart_test_loader.symbol_context, args_dict.dir_dataset)
    
    train_presence = semart_train_loader.symbol_context.sum(axis=0) > 0
    val_presence = semart_val_loader.symbol_context.sum(axis=0) > 0
    test_presence = semart_test_loader.symbol_context.sum(axis=0) > 0

    global_presence = train_presence * val_presence * test_presence

    train_symbol_mat = semart_train_loader.symbol_context[:, global_presence]
    val_symbol_mat = semart_val_loader.symbol_context[:, global_presence]
    test_symbol_mat = semart_test_loader.symbol_context[:, global_presence]

    canon_names = semart_train_loader.symbols_names    

    pd.DataFrame(train_symbol_mat).to_csv('Data/global_train_symbol_mat.csv')
    pd.DataFrame(val_symbol_mat).to_csv('Data/global_val_symbol_mat.csv')
    pd.DataFrame(test_symbol_mat).to_csv('Data/global_test_symbol_mat.csv')

    canon_names.iloc[global_presence].to_csv('Data/global_canon_names.csv')

    print('Hola')