import pickle
#from visdom import Visdom
import numpy as np
import csv
import pandas as pd


def save_obj(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    print("Saved object to %s." % filename)


def load_obj(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f, encoding='latin1')
    f.close()
    return obj

def load_csv_as_dict(csv_path):
    with open(csv_path, mode='r') as infile:
        reader = csv.reader(infile)
        
        mydict = {rows[1]:rows[0] for rows in reader}
    
    return mydict

def emd_to_csv(emd_path):
    name_csv = ''.join(emd_path.split('.')[:-1]) + '.csv'
    emd_data = pd.read_csv(emd_path, skiprows=1, sep=' ', header=None, index_col=0)
    emd_data = emd_data.sort_index()
    emd_data.columns = np.arange(emd_data.shape[1])


    emd_data.to_csv(name_csv)

def save_att_as_csv(data, csv_path):
    aux = pd.DataFrame.from_dict(data, orient='index')
    aux['key'] = aux.index
    aux.to_csv(csv_path, header=None, index=None)
    

#### GRAPH MATCHING AND COMPARISON
def graph_similarity(g1, g2):
    return 1 - np.abs(g1-g2).sum().sum() / (g1.shape[0] * g1.shape[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = None #Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

