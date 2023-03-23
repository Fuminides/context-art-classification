import torch
import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from gensim.models import Word2Vec
import text_encoding
from PIL import Image


class ArtDatasetMTL(data.Dataset):

    def __init__(self, args_dict, set, att2i, transform = None, clusters=15, k=100):
        """
        Args:
            args_dict: parameters dictionary
            set: 'train', 'val', 'test'
            att2i: list of attribute vocabularies as [type2idx, school2idx, time2idx, author2idx]
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
        self.type_vocab = att2i[0]
        self.school_vocab = att2i[1]
        self.time_vocab = att2i[2]
        self.author_vocab = att2i[3]

        self.imageurls = list(df['IMAGE_FILE'])
        self.type = list(df['TYPE'])
        self.school = list(df['SCHOOL'])
        self.time = list(df['TIMEFRAME'])
        self.author = list(df['AUTHOR'])

        self.embedds = args_dict.embedds
        # Load Data + Graph Embeddings
        self.graphEmb = []
        
        if self.embedds == 'graph':
            self.graphEm = Word2Vec.load(os.path.join(args_dict.dir_data, args_dict.graph_embs))
        elif self.embedds == 'bow':
            self.chosen_coded_semart_train, self.chosen_coded_semart_val, self.chosen_coded_semart_test = \
            text_encoding.bow_load_train_text_corpus(args_dict.dir_dataset, append='append', k=k)

            self.chosen_coded_semart_train = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_train, clusters=clusters)
            self.chosen_coded_semart_val = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_val, clusters=clusters)
            self.chosen_coded_semart_test = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_test, clusters=clusters)
            
            pd.DataFrame(self.chosen_coded_semart_train).to_csv('train_semart_fcm_'+str(clusters)+'.csv')
            pd.DataFrame(self.chosen_coded_semart_val).to_csv('val_semart_fcm_'+str(clusters)+'.csv')
            pd.DataFrame(self.chosen_coded_semart_test).to_csv('test_semart_fcm_'+str(clusters)+'.csv')
            
        elif self.embedds == 'frbc':
            self.chosen_coded_semart_train = pd.read_csv('Data/rule_embds_train.csv', index_col=0).values
            self.chosen_coded_semart_val = pd.read_csv('Data/rule_embds_val.csv', index_col=0).values
            self.chosen_coded_semart_test = pd.read_csv('Data/rule_embds_test.csv', index_col=0).values 
       

        elif self.embedds == 'tfidf':
            self.chosen_coded_semart_train, self.chosen_coded_semart_val, self.chosen_coded_semart_test = \
            text_encoding.tf_idf_load_train_text_corpus(args_dict.dir_dataset, append='append', k=k)

            self.chosen_coded_semart_train = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_train, clusters=clusters)
            self.chosen_coded_semart_val = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_val, clusters=clusters)
            self.chosen_coded_semart_test = text_encoding.fcm_coded_context(
                self.chosen_coded_semart_test, clusters=clusters)
        
        elif self.embedds == 'clip':
            self.chosen_coded_semart_train, self.chosen_coded_semart_val, self.chosen_coded_semart_test = \
            text_encoding.clip_load_train_text_corpus(args_dict.dir_dataset, append='append', k=k)


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
        type_idclass = self.class_from_name(self.type_vocab, self.type[index])
        school_idclass = self.class_from_name(self.school_vocab, self.school[index])
        time_idclass = self.class_from_name(self.time_vocab, self.time[index])
        author_idclass = self.class_from_name(self.author_vocab, self.author[index])


        

        # Graph embedding (only training samples)
        if self.set == 'train':
            if self.embedds == 'graph':
                graph_emb = self.graphEm.wv[self.imageurls[index]]
            else:
                graph_emb = self.chosen_coded_semart_train[index, :]

            graph_emb = torch.FloatTensor(graph_emb)


        elif self.set == 'val':
            if self.embedds == 'graph':
                graph_emb = self.graphEm.wv[self.imageurls[index]]
            else:
                graph_emb = self.chosen_coded_semart_val[index, :]

            graph_emb = torch.FloatTensor(graph_emb)

        
        elif self.set == 'test':
            if self.embedds == 'graph':
                    graph_emb = np.random.rand(128,1)#; self.graphEm.wv[self.imageurls[index]]
            else:
                graph_emb = self.chosen_coded_semart_test[index, :]

            graph_emb = torch.FloatTensor(graph_emb)


        if self.args_dict.att == 'all':
            return [image], [type_idclass, school_idclass, time_idclass, author_idclass, graph_emb]
        else:
            return [image], [idclass, graph_emb], self.imageurls[index]
