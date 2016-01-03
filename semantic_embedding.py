import os
from util.im2vec import Image2Vec
from util.query2vec import Query2Vec
from util.irc_image import calImageSimiByCos
from basic.common import ROOT_PATH
from simpleknn.bigfile import BigFile

class SemanticEmbedding:
    def __init__(self, label_source, corpus, word2vec_model, feat_path, rootpath=ROOT_PATH):
        label_vec_path = os.path.join('data', label_source, 'label_vec')
        label_id_file = os.path.join('data', label_source, 'label.txt')
        self.im2vec = Image2Vec(label_id_file, label_vec_path)
        self.qry2vec = Query2Vec(corpus, word2vec_model, rootpath)
        self.img_feats = BigFile(feat_path)

    def do_search(self, query, iid_list, k):
        # convert query to vector
        qvec = self.qry2vec.embedding(query)
        if qvec is not None:

            renamed, test_X = self.img_feats.read(iid_list)

            imgvecs = []
            for iid in iid_list:
                img_label = test_X[renamed.index(iid)]
                imgvecs.append(self.im2vec.embedding(img_label, k))

            scorelist = calImageSimiByCos(qvec, imgvecs)

        else:
            scorelist = []

        return scorelist