# coding: utf-8
import os
import numpy as np
from basic.common import printStatus
from simpleknn.bigfile import BigFile


DEFAULT_K = 10
DEFAULT_BLOCK_SIZE = 2000

INFO = os.path.basename(__file__)

class Image2Vec:

    def __init__(self, label_file, label2vec_dir):
        self.labels = map(str.strip, open(label_file).readlines())
        self.nr_of_labels = len(self.labels)
        feat_file = BigFile(label2vec_dir)
        renamed, vectors = feat_file.read(self.labels)
        name2index = dict(zip(renamed, range(len(renamed))))
        self.label_vectors = [None] * self.nr_of_labels
        self.feat_dim = feat_file.ndims

        for i in xrange(self.nr_of_labels):
            idx = name2index.get(self.labels[i], -1)
            self.label_vectors[i] = np.array(vectors[idx]) if idx >= 0 else None

        nr_of_inactive_labels = len([x for x in self.label_vectors if x is None])    
        printStatus(INFO, '#active_labels=%d, embedding_size=%d' % (self.nr_of_labels - nr_of_inactive_labels, self.feat_dim))



    def embedding(self, prob_vec, k=10):
        assert(len(prob_vec) == self.nr_of_labels), 'len(prob_vec)=%d, nr_of_labels=%d' % (len(prob_vec), self.nr_of_labels)
        top_hits = np.argsort(prob_vec)[::-1][:k]
        new_vec = np.array([0.] * self.feat_dim)

        Z = 0.
        for idx in top_hits:
            vec = self.label_vectors[idx]
            if vec is not None:
                new_vec += prob_vec[idx] * vec
                Z += prob_vec[idx]
        if Z > 1e-10:
            new_vec /= Z
        return new_vec