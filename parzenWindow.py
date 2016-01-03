import os
import sys
import collections
import numpy as np
# import mlpy

from util.tools import readQidQuery, writeRankingResult, writeDCGResult
from util.irc_image import calImageSimiByCos
from simpleknn.bigfile import BigFile
from basic.metric import getScorer
from basic.annotationtable import readAnnotationsFrom
from basic.common import ROOT_PATH, printMessage, checkToSkip, makedirsforfile


def calParzen(v, X, sig=20):
    v = np.array(v)
    X = np.array(X)
    diff = (v-X)
    sumlist = np.exp(- np.sum( diff * diff, 1) / (2 * sig * sig) )
    # return sum(sumlist)/(len(X)*sig*math.sqrt(2*math.pi))
    return sum(sumlist)/len(X)

# fast version based on mlpy
# def calParzen_fast(X, length, sig=0.5):
#     X = np.array(X)
#     sumlist = mlpy.kernel_gaussian(X,X,sigma=sig)
#     return np.sum(sumlist,0) / length



def process(options, collection):
    rootpath = options.rootpath
    overwrite = options.overwrite
    feature = options.feature
    method = options.method
    sigma =options.sigma

    # result path
    ranking_result_path = os.path.join(rootpath, collection, 'SimilarityIndex', collection, 'MetaData', method, feature)
    DCG_result_path = os.path.join(rootpath, collection, 'DCG', method, feature)
    if checkToSkip(ranking_result_path, overwrite):
        sys.exit(0)
    if checkToSkip(DCG_result_path, overwrite):
        sys.exit(0)
    
    # inpute of query
    qid_query_file = os.path.join(rootpath, collection, 'Annotations', 'qid.text.txt')
    qid_list, query_list = readQidQuery(qid_query_file)
    qid2query =  dict(zip(qid_list, query_list))
    
    # inpute of image
    img_feat_path = os.path.join(rootpath, collection, 'FeatureData', feature)
    img_feats = BigFile(img_feat_path)

    # the model to calculate DCG@25
    scorer = getScorer("DCG@25")


    done = 0
    qid2dcg = collections.OrderedDict()
    qid2iid_label_score = {}

    for qid in qid_list:
        iid_list, label_list = readAnnotationsFrom(collection, 'concepts%s.txt' % collection, qid, False, rootpath)

        renamed, test_X = img_feats.read(iid_list)

        parzen_list = []
        for imidx in iid_list:
            parzen_list.append(calParzen(img_feats.read_one(imidx), test_X , sigma))


        # parzen_list_suffle = calParzen_fast(test_X, len(renamed), sigma)
        # parzen_list = []
        # for imidx in iid_list:
        #     parzen_list.append(parzen_list_suffle[renamed.index(imidx)])


        sorted_tuple = sorted(zip(iid_list, label_list, parzen_list), key=lambda v:v[2], reverse=True)
        qid2iid_label_score[qid] = sorted_tuple

        # calculate DCG@25
        sorted_label = [x[1] for x in sorted_tuple]
        qid2dcg[qid] = scorer.score(sorted_label)
        printMessage("Done", qid, qid2query[qid])

        done += 1
        if done % 20 == 0:
             writeRankingResult(ranking_result_path, qid2iid_label_score)
             qid2iid_label_score = {}


    writeDCGResult(DCG_result_path, qid2dcg)
    writeRankingResult(ranking_result_path, qid2iid_label_score)
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
    parser = OptionParser(usage="""usage: %prog [options] collection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--feature", default="ruccaffefc7.imagenet", type="string", help="image feature (default: ruccaffefc7.imagenet)")
    parser.add_option("--method", default='pw', type="string", help="method (default: pw)")
    parser.add_option("--sigma", default='20', type="float", help="super parameter for parzen window")
       
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())    
