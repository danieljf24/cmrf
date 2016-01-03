
import cPickle
import theano
import numpy as np
import theano.tensor as T
from numpy import linalg as LA

from simpleknn.bigfile import BigFile
from util.query2vec import Query2Vec
from basic.common import ROOT_PATH

from model.devise_model import Devise
# from sklearn import preprocessing


class Devise_pre(object):
    def __init__(self, model_path, corpus, word2vec_model, feat_path, rootpath=ROOT_PATH):
        self.qry2vec = Query2Vec(corpus, word2vec_model, rootpath)
        self.img_feats = BigFile(feat_path)

        print model_path
        devise_model = cPickle.load(open(model_path, 'rb'))
        words_vec = T.matrix(dtype=theano.config.floatX)
        img_vec = T.matrix(dtype=theano.config.floatX)
        # compile a predictor function
        self.predict_model = theano.function(
            inputs=[words_vec, img_vec],
            outputs=devise_model.predict_score_one2many(words_vec, img_vec),
            allow_input_downcast=True)

    def predict_score(self, query, iid_list, normalization = 'L2'):

        qvec = self.qry2vec.embedding(query)
        
        
        if qvec is not None:

            # L2 normalization
            if normalization == 'L2':
                qvec = qvec / LA.norm(qvec,2)
            
            renamed, test_X = self.img_feats.read(iid_list)

            X = []
            for iid in iid_list:
                img_label = test_X[renamed.index(iid)]
                X.append(img_label)

            query_array = np.array([qvec])
            image_array = np.array(X)

            scorelist = self.predict_model(query_array, image_array)[0].tolist()
            # scorelist =  np.reshape(temp,(1,-1))[0].tolist()

        else:
            scorelist = []

        return scorelist