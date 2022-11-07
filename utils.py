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

#### DEEP FEATURES TO UNIQUE CSV
def format_features(path='./DeepFeatures/'):
    import os

    get_file_num = lambda a: int(a.split('_')[2])

    def get_max_number(path1):
        
        max_num = 0
        for file in os.listdir(path1):
            file_number = get_file_num(file)
            if file_number > max_num:
                max_num = file_number
        
        return max_num
    
    def get_X(path1, dataset='train'):
        ix = 0
        for file in os.listdir(path1):
            if (file.split('_')[1] == 'x') and (file.split('_')[0] == dataset):
                x_file = pd.read_csv(path + file, index_col=0)

                if ix == 0:
                    res = x_file
                else:
                    res = pd.concat([res, x_file])
        
        return res

    def get_y(path1, dataset, task):
        ix = 0
        for file in os.listdir(path1):
            if (file.split('_')[1] == 'y') and (file.split('_')[-1].split('.')[0] == task) and (file.split('_')[0] == dataset):
                x_file = pd.read_csv(path + file)

                if ix == 0:
                    res = x_file
                else:
                    res = pd.concat([res, x_file])
                    ix=+1
        
        return res

    X = get_X(path, 'train')
    X_test = get_X(path, 'test')

    y_author = get_y(path, 'train', 'author')
    y_type = get_y(path, 'train', 'type')
    y_time = get_y(path, 'train', 'time')
    y_school = get_y(path, 'train', 'school')

    y_author_test = get_y(path, 'test', 'author')
    y_type_test = get_y(path, 'test', 'type')
    y_time_test = get_y(path, 'test', 'time')
    y_school_test = get_y(path, 'test', 'school')


    y_final = pd.DataFrame(np.zeros((y_type.iloc[:, 0].shape[0], 4)))
    y_final.iloc[:, 0] = y_type.iloc[:, 1]
    y_final.iloc[:, 1] = y_time.iloc[:, 1]
    y_final.iloc[:, 2] = y_author.iloc[:, 1]
    y_final.iloc[:, 3] = y_school.iloc[:, 1]

    y_final_test = pd.DataFrame(np.zeros((y_type_test.iloc[:, 0].shape[0], 4)))
    y_final_test.iloc[:, 0] = y_type_test.iloc[:, 1]
    y_final_test.iloc[:, 1] = y_time_test.iloc[:, 1]
    y_final_test.iloc[:, 2] = y_author_test.iloc[:, 1]
    y_final_test.iloc[:, 3] = y_school_test.iloc[:, 1]


    return X, y_final, X_test, y_final_test
    
def performance_classifier_tasks(clf, path):
    X_train, y_train, X_test, y_test = format_features(path)
    tasks = ['Type', 'Time', 'Author', 'School']

    for task in range(4):
        clf.fit(X_train, y_train.to_numpy()[:,task])
        print('Task train performance ' + tasks[task] +': '+ str(np.mean(np.equal(clf.predict(X_train), y_train.to_numpy()[:,task]))))
        print('Task test performance ' + tasks[task] +': '+ str(np.mean(np.equal(clf.predict(X_test), y_test.to_numpy()[:,task]))))

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

if __name__ == '__main__':
    path = 'C:/Users/jf22881/Downloads/Clip_features/DeepFeatures/'
    from sklearn import svm
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=0)
    performance_classifier_tasks(clf, path)
    