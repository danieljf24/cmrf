import os
import numpy as np

from simpleknn.bigfile import BigFile
from basic.common import ROOT_PATH

class Query2Vec:

    def __init__(self, corpus, modelName, rootpath=ROOT_PATH):
        word2vec_dir = os.path.join(rootpath, "word2vec", corpus, modelName)
        self.word2vec = BigFile(word2vec_dir)

    def get_feat_dim(self):
        return self.word2vec.ndims

    def mapping(self, query):
            #words = query.lower().split(',')
            words = query.lower().split('/')
            res = []
            for word in words:
                res += word.strip().replace('_', ' ').split()

            word_vecs = []
            newname = []
            for w in res:
                renamed, vectors = self.word2vec.read([w])
                if vectors: 
                    word_vecs.append(vectors[0])
                    newname.append(renamed[0])

            #print wnid, res, len(word_vecs)
            if len(word_vecs)>0:
                return np.array(word_vecs).mean(axis=0)
            else:
                return None
                
    def embedding(self, wnid):
        return self.mapping(wnid)