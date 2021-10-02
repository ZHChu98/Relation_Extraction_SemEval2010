import numpy as np
import os
import math
import string
from tqdm import tqdm

def read_data_sets(noZero=False):
    if os.path.exists('word2vec.npy') == False:
        return
    
    sents = []
    rels = []
    
    fi = open("data/TRAIN_FILE_splited.txt", 'r')
    for i in tqdm(range(8000)):
        sent = fi.readline()
        sent = sent.replace('<e1> ', '').replace(' </e1>', '').replace('<e2> ', '').replace(' </e2>', '')
        sent = sent.rstrip('\n').rstrip(' ')
        sent = sent.split(' ')
        rel = fi.readline()
        rel = int(rel)
        
        if noZero==True and rel==18:
            continue
        sents.append(sent)
        rels.append(rel)
    fi.close()
    
    fi = open("data/TEST_FILE_splited.txt", 'r')
    for i in tqdm(range(2717)):
        sent = fi.readline()
        sent = sent.replace('<e1> ', '').replace(' </e1>', '').replace('<e2> ', '').replace(' </e2>', '')
        sent = sent.rstrip('\n').rstrip(' ')
        sent = sent.split(' ')
        rel = fi.readline()
        rel = int(rel)
        
        if noZero==True and rel==0:
            continue
        sents.append(sent)
        rels.append(rel)
    fi.close()
    
    rels = np.array(rels)
    semdata = SemData(sents, rels, noZero)
    return semdata

    
class Data(object):
    def __init__(self, sents, rels, embeddings_dict, noZero):
        n_embedding = 200
        self.sents = []
        self.pos = 0

        for i in range(len(rels)):
            sent = [embeddings_dict[word] for word in sents[i]]
            self.sents.append(np.array(sent))
        
        if noZero==True:
            self.rels = np.eye(18)[rels]
        else:
            self.rels = np.eye(19)[rels]
        
        
    def next_batch(self, batch_size):
        if self.pos+batch_size > len(self.sents):
            self.pos = self.pos + batch_size - len(self.sents)
        res = (self.sents[self.pos:self.pos+batch_size], self.rels[self.pos:self.pos+batch_size])
        self.pos += batch_size
        return res
    
    
class SemData(object):
    def __init__(self, sents, rels, noZero):
        n_class = 19 - noZero
        embeddings = np.load('word2vec.npy').item()
        self.embeddings = np.array(list(embeddings.values()))
        embeddings_dict = dict()
        i = 0
        for word in list(embeddings.keys()):
            embeddings_dict[word] = i
            i += 1
        
        self.train = Data(sents[0:8000], rels[0:8000], embeddings_dict, noZero)
        self.test = Data(sents[8000:], rels[8000:], embeddings_dict, noZero)
        self.weights = np.array([len(rels)/sum(rels==i) for i in range(n_class)])
    

def distribution(labels, train_size, noZero):
    fp = open('distribution.txt', 'w')
    n_class = 19 - noZero
    
    dist = [sum(labels==i) for i in range(n_class)]
    fp.write('all data\n')
    for i in range(n_class):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/len(labels)))
    dist = [sum(labels[0:train_size]==i) for i in range(n_class)]
    fp.write('\ntrain data\n')
    for i in range(n_class):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/train_size))
    dist = [sum(labels[train_size:]==i) for i in range(n_class)]
    fp.write('\ntest data\n')
    for i in range(n_class):
        fp.write('%2d: %5d  %2.2f %%\n' % (i+noZero, dist[i], 100*dist[i]/(len(labels)-train_size)))
        
    fp.close()