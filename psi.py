
import cPickle
import theano
import numpy as np
import theano.tensor as T

from simpleknn.bigfile import BigFile
from util.query2vec import Query2Vec
from basic.common import ROOT_PATH

from theano import sparse
from model.psi_model import PSI
# from sklearn import preprocessing


class PSI_pre(object):
    def __init__(self, model_path, bow_path, feat_path):
        # voabulary_file = os.path.join('result', 'msr2013train_voabulary_query_bow.pkl')
        self.count_vect, self.tf_transformer = cPickle.load(open(bow_path, 'rb'))
        self.img_feats = BigFile(feat_path)

        # print model_path
        devise_model = cPickle.load(open(model_path, 'rb'))
        # words_vec = T.matrix(dtype=theano.config.floatX)
        words_vec = sparse.csr_matrix(dtype=theano.config.floatX)
        img_vec = T.matrix(dtype=theano.config.floatX)
        # compile a predictor function
        self.predict_model = theano.function(
            inputs=[words_vec, img_vec],
            outputs=devise_model.predict_score_one2many(words_vec, img_vec),
            allow_input_downcast=True)

    def predict_score(self, query, iid_list):

        test_counts = self.count_vect.transform([query])
        query_vec = self.tf_transformer.transform(test_counts)
        # print query_vec
        # print query_vec.shape
        
        # if qvec is not None:
        renamed, test_X = self.img_feats.read(iid_list)

        X = []
        for iid in iid_list:
            img_label = test_X[renamed.index(iid)]
            X.append(img_label)

        image_array = np.array(X)


        temp = self.predict_model(query_vec, image_array)
        scorelist =  np.reshape(temp,(1,-1))[0].tolist()

        return scorelist