import os
import sys
import timeit
import random
import numpy
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
from theano import sparse
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.nlinalg import diag
from theano.ifelse import ifelse

from simpleknn.bigfile import BigFile
from basic.common import ROOT_PATH, makedirsforfile, checkToSkip
import cPickle
from numpy import linalg as LA
# from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

INFO = os.path.basename(__file__)

DEFAULT_CORPUS = 'flickr'
DEFAULT_WORD2VEC = 'vec200flickr25m'
DEFAULT_EMBEDDING = 'equal'

class PSI(object):

    def __init__(
        self,
        numpy_rng,
        input_plus=None,
        input_minus=None,
        input_query=None,
        click=None,
        n_img=4096,
        n_query=50000,
        n_hidden=200,
        W_x=None,
        W_q=None
    ):
       
        self.n_img = n_img
        self.n_query = n_query


        if not W_x:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_img + n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_img + n_hidden)),
                    size=(n_img, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # initial_W = numpy.zeros(
            #     (n_input, n_output),
            #     dtype=theano.config.floatX
            # )
            W_x = theano.shared(value=initial_W, name='W_x', borrow=True)


        if not W_q:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_query + n_hidden)),
                    high=4 * numpy.sqrt(6. / (n_query + n_hidden)),
                    size=(n_query, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W_q = theano.shared(value=initial_W, name='W_q', borrow=True)


        self.W_x = W_x
        self.W_q = W_q


        if input_plus is None:
            self.plus = T.dmatrix(name='plus')
        else:
            self.plus = input_plus

        if input_minus is None:
            self.minus = T.dmatrix(name='minus')
        else:
            self.minus = input_minus
        
        if input_query is None:
            self.query = T.dmatrix(name='query')
        else:
            self.query = input_query


        self.params = [self.W_x, self.W_q]


    def predict_img(self, input):
        return T.dot(input, self.W_x)

    def predict_query(self, input):
        return sparse.basic.dot(input, self.W_q)
        # return T.switch(output > 0, output, 0)   # T.nnet.relu


    def predict_score_batch(self, query, image):
        query_matrix = sparse.basic.dot(query, self.W_q)
        query_matrix = query_matrix.dimshuffle(1,0)

        img_matrix = T.dot(T.dot(image, self.W_x), query_matrix)
        return diag(img_matrix)

    def predict_score_one2many(self, query, image):
        query_matrix = sparse.basic.dot(query, self.W_q)
        query_matrix = query_matrix.dimshuffle(1,0)

        img_matrix = T.dot(T.dot(image, self.W_x), query_matrix)
        return img_matrix


    def get_cost(self):
        query_matrix = sparse.basic.dot(self.query, self.W_q)
        query_matrix = query_matrix.dimshuffle(1,0)

        plus_matrix = T.dot(T.dot(self.plus, self.W_x), query_matrix)
        plus_vector = diag(plus_matrix)

        minus_matrix = T.dot(T.dot(self.minus, self.W_x), query_matrix)
        minus_vector = diag(minus_matrix)

        cost_vector = T.maximum(0.0, 1 - plus_vector + minus_vector)

        return T.mean(cost_vector)


class DataSet():
    def __init__(self, data_file, batch_size):
        img_plus_list = []
        img_minus_list = []
        query_list = []
        click_list = []
        for line in open(data_file):
            data = line.strip().split('\t')
            img_plus_list.append(data[0])
            img_minus_list.append(data[1])
            query_list.append(data[2])

        self.img_plus_list = img_plus_list
        self.img_minus_list = img_minus_list
        self.query_list = query_list
        assert len(img_plus_list) == len(img_minus_list)
        assert len(img_minus_list) == len(query_list)

        self.datasize = len(img_plus_list)
        self.batch_size = batch_size


        # fout_file = os.path.join('result', 'msr2013train_query_bow_1.pkl')
        # self.count_vect, self.tf_transformer = cPickle.load(open(fout_file, 'rb'))
        # print "start loagind data..."
        voabulary_file = os.path.join('result', 'msr2013train_voabulary_query_bow.pkl')
        self.voabulary_bow = cPickle.load(open(voabulary_file, 'rb'))
        # print "data set have been initialized"



    def getBatchData(self, index, img_feats):

        plus_list = self.img_plus_list[index*self.batch_size: (index+1)*self.batch_size]
        minus_list = self.img_minus_list[index*self.batch_size: (index+1)*self.batch_size]
        query_list = self.query_list[index*self.batch_size: (index+1)*self.batch_size]

        plus = []
        renamed, feats = img_feats.read(plus_list)
        for img in plus_list:
            plus.append(feats[renamed.index(img)])

        minus = []
        renamed, feats = img_feats.read(minus_list)
        for img in minus_list:
            minus.append(feats[renamed.index(img)])

        query = self.voabulary_bow[query_list, :]

        # relevant images, irrelevant image, query of word2vec   
        return (plus, minus, query)





def process(options, collection):

    rootpath = options.rootpath
    overwrite =  options.overwrite

    corpus = options.corpus
    word2vec = options.word2vec
    embedding = options.embedding

    query_feature = options.query_feature
    img_feature = options.img_feature

    batch_size = options.batch_size
    initial_learning_rate = options.learning_rate
    learning_rate_decay = options.learning_rate_decay

    init_model = options.init_model_from
    checkpoint_output_dir = options.checkpoint_output_dir

    # img2vec
    img_feat_path = os.path.join(rootpath, collection, 'FeatureData', img_feature)
    img_feats = BigFile(img_feat_path)

    # dataset 
    train_file = os.path.join(rootpath, collection, 'TextData', 'nlp',  'mincc_2_maximg_20_for_bow', 'train.triplet.random.bow.txt')
    val_file = os.path.join(rootpath, collection, 'TextData', 'nlp',  'mincc_2_maximg_20_for_bow','val.triplet.random.bow.txt')
    
    # log file
    model_file = os.path.join(rootpath, collection, 'cv' ,checkpoint_output_dir,  
         query_feature + '_' + img_feature, INFO, 'lr_%.4f' % initial_learning_rate)
    
    if checkToSkip(model_file, overwrite):
        sys.exit(0)
    try:
        os.makedirs(model_file)
    except:
        pass

    # training and validation set
    trainData = DataSet(train_file, batch_size)
    validData = DataSet(val_file, batch_size)

    # compute number of minibatches for training, validation and testing
    n_train_batches = trainData.datasize / batch_size
    n_valid_batches = validData.datasize / batch_size



    x = T.matrix('x', dtype='float32')  # the data is presented as rasterized images'
    y = T.matrix('y', dtype='float32')
    # q = T.matrix('q')
    q = sparse.csr_matrix(name='q', dtype='float32')
    newX = T.matrix(dtype=x.dtype)
    newY = T.matrix(dtype=y.dtype)
    # newQ = T.matrix(dtype=q.dtype)
    newQ = sparse.csr_matrix(dtype=q.dtype)

    epoch = T.scalar()
    iter_max_iter = T.scalar()

    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each batch instead
    decay_learning_rate = theano.function(inputs=[iter_max_iter], outputs=learning_rate,
            updates={learning_rate:  initial_learning_rate * (1 - iter_max_iter)**2 })
   
    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #         updates={learning_rate: learning_rate * learning_rate_decay})


    #### the params for momentum
    mom_start = 0.5
    mom_end = 0.99
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 20
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}


    ######################
    # BUILDING THE MODEL #
    ######################

    rng = numpy.random.RandomState(123)

    if init_model != '':
        psi_init = cPickle.load(open(init_model, 'rb'))
        psi = PSI(
            numpy_rng=rng,
            input_plus=x,
            input_minus=y,
            input_query=q,
            n_img=options.n_img,
            n_query=options.n_query,
            n_hidden=options.n_hidden,
            W_x=cca_click_init.W_x,
            W_q=psi_init.W_q
        )
    else:
        psi = PSI(
            numpy_rng=rng,
            input_plus=x,
            input_minus=y,
            input_query=q,
            n_img=options.n_img,
            n_query=options.n_query,
            n_hidden=options.n_hidden
        )


    cost = psi.get_cost()

    # the optimization code refer to https://github.com/mdenil/dropout
    # Compute gradients of the model wrt parameters
    gparams = []
    for param in psi.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # ... and allocate mmeory for momentum'd versions of the gradient
    gparams_mom = []
    for param in psi.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)

    # Update the step direction using momentum
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        # Misha Denil's original version
        #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
      
        # change the update rule to match Hinton's dropout paper
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(psi.params, gparams_mom):
        # Misha Denil's original version
        #stepped_param = param - learning_rate * updates[gparam_mom]
        
        # since we have included learning_rate in gparam_mom, we don't need it
        # here
        stepped_param = param + updates[gparam_mom]

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        # if param.get_value(borrow=True).ndim == 2:
        #     #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
        #     #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
        #     #updates[param] = stepped_param * scale
            
        #     # constrain the norms of the COLUMNs of the weight, according to
        #     # https://github.com/BVLC/caffe/issues/109
        #     col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
        #     desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
        #     scale = desired_norms / (1e-7 + col_norms)
        #     updates[param] = stepped_param * scale
        # else:
        #     updates[param] = stepped_param
        updates[param] = stepped_param



    train_psi = theano.function(
        [newX, newY, newQ, epoch],
        cost,
        updates=updates,
        givens={
            x : newX,
            y : newY,
            q : newQ
        },
        allow_input_downcast=True
    )


    validate_psi = theano.function(
        [newX, newY, newQ],
        cost,
        givens={
            x : newX,
            y : newY,
            q : newQ
        },
        allow_input_downcast=True
    )


    ###############
    # TRAIN MODEL #
    ###############

    best_validation_loss = numpy.inf
    c_iter = 0
    best_iter = 0
    epoch_counter = 0
    early_stop_counter = 0
    start_time = timeit.default_timer()
    max_epochs = options.max_epochs
    print 'training...'
    while (epoch_counter < max_epochs):
        epoch_counter = epoch_counter + 1
        training_losses = []
        t0 = timeit.default_timer()
        index_list = range(n_train_batches)
        random.shuffle(index_list)
        for minibatch_index in index_list:
            new_train_X, new_train_Y, new_train_Q = trainData.getBatchData(minibatch_index, img_feats)
            training_losses.append(train_psi(new_train_X, new_train_Y, new_train_Q, epoch_counter))
           

            # iteration number
            c_iter += 1
            new_learning_rate = decay_learning_rate(1.0*c_iter/(max_epochs*n_train_batches))

            if c_iter % options.log_step == 0:
                t1 = timeit.default_timer()
                print '%d/%d, cost: %f, learning rate: %.10f, time: %.3fs' % (c_iter, max_epochs*n_train_batches, numpy.mean(training_losses), learning_rate.get_value(), (t1-t0))
                training_losses = []
                t0 = timeit.default_timer()



        print "validation..."
        validation_losses = []
        for i in range(n_valid_batches):
            new_val_X, new_val_Y, new_val_Q = validData.getBatchData(i, img_feats)
            validation_losses.append(validate_psi(new_val_X, new_val_Y, new_val_Q))

        this_validation_loss = numpy.mean(validation_losses)

        print(
            'epoch %i, minibatch %i/%i, validation error %f ' %
            (
                epoch_counter,
                minibatch_index + 1,
                n_train_batches,
                this_validation_loss
            )
        )
        
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:

            best_validation_loss = this_validation_loss
            best_iter = c_iter

            # write out the parameters of psi modle
            fout_file = os.path.join(model_file, 'psi_lr_%.6f_iter_%d_loss_%.3f.pkl' % ( learning_rate.get_value(), best_iter, best_validation_loss))                  
            fout = file(fout_file, 'wb')
            cPickle.dump(psi, fout, -1)
            fout.close()
        else:
            # do early stopping when loss not increase on validation set on three consecutive times 
            early_stop_counter += 1
            if early_stop_counter > 3:
                break


    end_time = timeit.default_timer()
    print ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))




def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)

    parser.add_option('-o', '--checkpoint_output_dir', type="string", default='cv_%s' % INFO, help='output directory to write checkpoints to')
    parser.add_option('--write_checkpoint_ppl_threshold', type="float", default=-1, help='ppl threshold above which we dont bother writing a checkpoint to save space')
    parser.add_option('--init_model_from', type="string", default='', help='initialize the model parameters from some specific checkpoint?')

    # word2vector parameters
    parser.add_option("--corpus", default=DEFAULT_CORPUS, type="string", help="corpus using which word2vec is trained (default: %s)" % DEFAULT_CORPUS)
    parser.add_option("--word2vec", default=DEFAULT_WORD2VEC, type="string", help="word2vec model (default: %s)" % DEFAULT_WORD2VEC)
    parser.add_option("--embedding", default=DEFAULT_EMBEDDING, type="string", help="embedding model (default: %s)" % DEFAULT_EMBEDDING)
    
    # feature parameters
    parser.add_option("--query_feature", default="flickr_vec200flickr25m", type="string", help="query feature (default: flickr_vec200flickr25m)")
    parser.add_option("--img_feature", default="ruccaffefc7.imagenet", type="string", help="image feature (default: ruccaffefc7.imagenet)")

    # model parameters
    parser.add_option('--n_query', type="int", default=50000, help='the number of query input neurons')
    parser.add_option('--n_img', type="int", default=4096, help='the number of image input neurons')
    parser.add_option('--n_hidden', type="int", default=200, help='the number of n_hidden neurons')
  
    # optimization parameters
    parser.add_option('--max_epochs', type="int", default=50, help='number of epochs to train for')
    parser.add_option('--L1_reg', type="float", default=0.00, help='coefficient of L1 regulariation term')
    parser.add_option('--L2_reg', type="float", default=0.1, help='coefficient of L2 regulariation term(un-normalize: 0.1  normalize:   )')
    parser.add_option('--learning_rate', type="float", default=0.01, help='solver learning rate' )
    parser.add_option('--lr_gamma', type="float", default=0.5, help='drop the learning rate by a factor of 2') 
    parser.add_option('--momentum', type="float", default=0.0, help='momentum for vanilla sgd ( optimal: 0.9)')
    parser.add_option('--batch_size', type="int", default=100, help='batch size')
    parser.add_option('--learning_rate_decay', type="float", default=0.85, help='drop the learning rate by a factor of 2') 

     # evaluation parameters
    parser.add_option('--eval_period', type="float", default=0.1, help='in units of epochs, how often do we evaluate on val set?')
    parser.add_option('--improvement_thres', type="float", default=0.995, help='a relative improvement of this much is considered significant')
    parser.add_option('--log_step', type="int", default=10, help='display the running time log per log_step')
    

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())

