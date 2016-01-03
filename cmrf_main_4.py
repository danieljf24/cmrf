import os
import sys
import random
import collections

from image2text import Image2Text
from text2image import Text2Image
from semantic_embedding import SemanticEmbedding

from util.queryParser import SimpleQueryParser
from util.tools import readQidQuery, writeRankingResult, writeDCGResult

from simpleknn.bigfile import BigFile
from basic.metric import getScorer
from basic.annotationtable import readAnnotationsFrom
from basic.common import ROOT_PATH, printMessage, checkToSkip, makedirsforfile


def process(options, trainCollection, devCollection):
    rootpath = options.rootpath
    overwrite = options.overwrite
    method = options.method
    metric = options.metric

    qrysim = options.qrysim
    qrythres = options.qrythres
    ntopimg = options.ntopimg
    ntopqry = options.ntopqry
    mincc = options.mincc
    feature = options.feature

    # semantic embedding
    k = options.k
    corpus = options.corpus
    word2vec_model = options.word2vec
    label_source = options.label_source

    # result path
    ranking_result_path = os.path.join(rootpath, devCollection, 'SimilarityIndex', devCollection, 'MetaData', method, feature)
    DCG_result_path = os.path.join(rootpath, devCollection, metric, method, feature)
    if checkToSkip(ranking_result_path, overwrite):
        sys.exit(0)
    if checkToSkip(DCG_result_path, overwrite):
        sys.exit(0)

    # inpute of query
    qp = SimpleQueryParser()
    qid_query_file = os.path.join(rootpath, devCollection, 'Annotations', 'qid.text.txt')
    qid_list, query_list = readQidQuery(qid_query_file)   #(qid query)
    qid2query =  dict(zip(qid_list, [qp.process(query) for query in query_list]))
    
    # path of image feature
    train_feat_path = os.path.join(rootpath, trainCollection, 'FeatureData', feature)
    dev_feat_path = os.path.join(rootpath, devCollection, 'FeatureData', feature)


    # method selection
    if method =='conse':
        se_searcher = SemanticEmbedding(label_source, corpus, word2vec_model, dev_feat_path, rootpath)

    elif method == 't2i' or method == 'ta': 
        nnquery_file = os.path.join(rootpath, devCollection, 'TextData','querynn', options.nnqueryfile)
        qryClick_file = os.path.join(rootpath, trainCollection, 'TextData', options.queryclickfile)
        t2i_searcher = Text2Image(nnquery_file, qryClick_file, dev_feat_path, train_feat_path, ntopqry)

    elif method == 'i2t' or method == 'ia':
        nnimage_file = os.path.join(rootpath, devCollection, 'TextData','imagenn', feature, options.nnimagefile)
        imgClick_file = os.path.join(rootpath, trainCollection, 'TextData', options.imageclickfile)
        i2t_searcher = Image2Text(nnimage_file, imgClick_file, qrysim, ntopimg, ntopqry)

    else:
        print "this model is not supported with %s" % method
        sys.exit(0)


 
    # calculate DCG@25
    scorer = getScorer(metric)

    done = 0
    failed_count = 0
    qid2dcg = collections.OrderedDict()
    qid2iid_label_score = {}

    for query_id in qid_list:

        iid_list, label_list = readAnnotationsFrom(devCollection, 'concepts%s.txt' % devCollection, query_id, False, rootpath)        

        if method == 'conse':
            scorelist = se_searcher.do_search(qid2query[query_id], iid_list, k)

        elif method == 't2i':
            scorelist = t2i_searcher.text2image(query_id, iid_list, qrythres, mincc )

        elif method == 'ta':
            scorelist = t2i_searcher.textAnnotation( query_id, iid_list, ntopimg, qrythres, mincc)

        elif method == 'i2t': 
            scorelist = i2t_searcher.image2text(qid2query[query_id], iid_list, mincc )

        elif method == 'ia':
            scorelist = i2t_searcher.imageAnnotation( qid2query[query_id], iid_list, mincc )    
         

        if len(scorelist) == 0: 
            failed_count += 1
            scorelist = [0]*len(iid_list)
            qid2iid_label_score[query_id] = zip(iid_list, label_list, scorelist)
            random.shuffle(qid2iid_label_score[query_id])
        else:
            qid2iid_label_score[query_id] = zip(iid_list, label_list, scorelist)
            qid2iid_label_score[query_id] = sorted(qid2iid_label_score[query_id], key=lambda v:v[2], reverse=True)


        # calculate the result ranking of DCG@25 from our model
        qid2dcg[query_id] = scorer.score([x[1] for x in qid2iid_label_score[query_id]])
        printMessage("Done", query_id, qid2query[query_id])

        done += 1
        if(done % 20 == 0):
            writeRankingResult(ranking_result_path, qid2iid_label_score)
            qid2iid_label_score = {}
    
    writeRankingResult(ranking_result_path, qid2iid_label_score)
    writeDCGResult(DCG_result_path, qid2dcg)
    print "number of failed query: %d" % failed_count 
    print "average DCG@25: %f" % (1.0*sum(qid2dcg.values())/ len(qid2dcg.values()))

    result_path_file = "result/individual_result_pathes.txt"
    if os.path.exists(result_path_file):
        fout = open(result_path_file,'a')
    else:
        makedirsforfile(result_path_file)
        fout = open(result_path_file, 'w')
    fout.write(ranking_result_path + '\n')
    fout.close()
    


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] trainCollection devCollection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--method", default='conse', type="string", help="method:conse, t2i, i2t  (default: conse)")
    parser.add_option("--metric", default='DCG@25', type="string", help="metric (default: DCG@25)")
    parser.add_option("--feature", default="ruccaffefc7.imagenet", type="string", help="image feature (default: ruccaffefc7.imagenet)")

    # semantic embedding parameters
    parser.add_option("--k", default=10, type="int", help="top-k labels used for semantic embedding (default: 10)")
    parser.add_option("--corpus", default='flickr', type="string", help="corpus that word2vec model trained on (default: flickr)" )
    parser.add_option("--word2vec", default='vec200flickr25m', type="string", help="word2vec model (default: vec200flickr25m)" )
    parser.add_option("--label_source", default="ilsvrc12", type="string", help="image lable source for semantic embedding")

    # image2text parameters
    parser.add_option("--qrysim", default='O', type="string", help="query similarity method (default: O )")
    parser.add_option("--nnimagefile", default='id.100nn.dis.txt', type="string", help="top 100 visual neighbours with distance for each image")
    parser.add_option("--imageclickfile", default='image.clicked.txt', type="string", help="clicked data for each image")

    # text2image parameters
    parser.add_option("--nnqueryfile", default='qid.100nn.score.txt', type="string", help="top 100 visual neighbours with similarity score for each image")
    parser.add_option("--queryclickfile", default='query.clicked.txt', type="string", help="clicked data for each query")
    
    # image2text and text2image parameters
    parser.add_option("--qrythres", default=0.3, type="float", help="query similarity threshold (default: 0.3)")
    parser.add_option("--ntopimg", default=50, type="int", help="the number of top similar images (default: 50)")
    parser.add_option("--ntopqry", default=30, type="int", help="the number of top relevant queris (default: 10)")
    parser.add_option("--mincc", default=1, type="int", help="minimum click count (default: 1)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())    
