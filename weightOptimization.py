import os
import sys
import random
import numpy as np
from basic.metric import getScorer
from basic.common import checkToSkip, makedirsforfile

EPSILON = 1e-8

def load_data(filename='result/qid.img.lable.feature.txt'):
    res = []
    qid = None
    labels = []
    vecs = []
    
    for line in open(filename):
        elems = line.strip().split()
        if qid != elems[0]: # meet a new query
            if len(labels)>0:
                nr_of_images = len(labels)
                feat_dim = len(vecs[0])
                scores = np.zeros((nr_of_images, feat_dim))
                for i in range(nr_of_images):
                    scores[i,:] = vecs[i]
                res.append((qid, labels, scores))
                labels = []
                vecs = []
            qid = elems[0]
            
        labels.append(int(elems[2]))
        vecs.append(map(float, elems[3:]))
    
    scores = np.zeros((len(labels), len(vecs[0])))
    for i in range(len(labels)):
        scores[i,:] = vecs[i]
    res.append((qid, labels, scores))
    return res    


def eval(data, weights):
    scorer = getScorer('DCG@25')
    feat_dim = data[0][-1].shape[1]
    nr_of_qry = len(data)

    mean_perf = [0] * feat_dim
    avg_perf = 0

    for i in range(nr_of_qry):
        qid, labels, scores = data[i]
        fusion = scores.dot(weights)
        sorted_idx = (0-fusion).argsort()
        sorted_labels = [labels[x] for x in sorted_idx]
        avg_perf += scorer.score(sorted_labels)
    return avg_perf / nr_of_qry


def eval_file(data, weights, file_name):
    scorer = getScorer('DCG')
    feat_dim = data[0][-1].shape[1]
    nr_of_qry = len(data)

    mean_perf = [0] * feat_dim
    avg_perf = 0

    fout = open(file_name, 'w')

    for i in range(nr_of_qry):
        qid, labels, scores = data[i]
        fusion = scores.dot(weights)

        print '.' * 20
        print fusion
        print '-' * 20
        sorted_idx = (0-fusion).argsort()
        sorted_labels = [labels[x] for x in sorted_idx]
        fout.write(qid +" "+ str(scorer.score(sorted_labels)) + "\n") 
    fout.close()
    
                
def coordinate_ascent(data):
    feat_dim = data[0][-1].shape[1]
    alpha = [1.0/feat_dim] * feat_dim
    changed = [1] * feat_dim
    #random.seed()
    #activeIndex = random.randint(0, feat_dim-1)
    activeIndex = feat_dim-1
    
    while sum(changed) > 0:
        activeIndex = (activeIndex+1)% feat_dim
        max_obj = eval(data, alpha)
        obj_0 = max_obj
        best_weight = alpha[activeIndex]
        
        # bi-direction search                
        for sign in [1,-1]:
            if alpha[activeIndex]<EPSILON and (-1==sign): # no need to search in the descent direction
                continue    
            if 1 == sign:
                step = alpha[activeIndex] * 2
                if step < EPSILON:
                    step = 0.05
            else:
                step = alpha[activeIndex]                
            while step > EPSILON:
                delta = sign*step
                temp_alpha = list(alpha)
                temp_alpha[activeIndex] += delta
                obj = eval(data, temp_alpha)

                if max_obj < obj:
                    max_obj = obj
                    best_weight = alpha[activeIndex] + delta
                step /= 2                
        if abs(obj_0-max_obj)>EPSILON:
            print 'alpha[%2d]=%.4f -> %.4f,  obj (%.6f -> %.6f)' % (activeIndex, alpha[activeIndex], best_weight, obj_0, max_obj)
            alpha[activeIndex] = max(0,best_weight)
            Z = float(sum(alpha))
            alpha = [x/Z for x in alpha]
            changed[activeIndex] = 1
        else:
            print 'alpha[%2d]=%.4f unchanged, obj (%.6f -> %.6f)' % (activeIndex, alpha[activeIndex], obj_0, max_obj)            
            changed[activeIndex] = 0 

    return max_obj, alpha
                

def test():
    scorer = getScorer('DCG@25')
    data = load_data()
    feat_dim = data[0][-1].shape[1]
    nr_of_qry = len(data)


    mean_perf = [0] * feat_dim
    rand_perf = 0
    avg_perf = 0

    weights = [1.0/feat_dim] * feat_dim
        
    for i in range(nr_of_qry):
        qid, labels, scores = data[i]
        nr_of_img = len(labels)
        A = (0-scores).argsort(axis=0)
        for j in range(feat_dim):
            sorted_idx = A[:,j]
            sorted_labels = [labels[x] for x in sorted_idx]
            random_guess = scorer.score(random.sample(labels, len(labels)))
            run = scorer.score(sorted_labels)
            mean_perf[j] += run
            rand_perf += random_guess
        #print int(run > random_guess), run
        
        avg = scores.dot(weights)
        sorted_idx = (0-avg).argsort()
        sorted_labels = [labels[x] for x in sorted_idx]
        avg_perf += scorer.score(sorted_labels)
    
    print 'random guess', rand_perf / (feat_dim * nr_of_qry)
    for j in range(feat_dim):
        print 'DCG@25 of feature_%d: %s' % (j, mean_perf[j] / nr_of_qry)
    # print 'mean DCG@25 of different feature:' , avg_perf / nr_of_qry
        
    print 'DCG@25 of average_fusion:', eval(data, weights)

    

def process(options):

    overwrite = options.overwrite
    inputeFile = options.inputeFile
    weightFile = options.weightFile


    weightFile = os.path.join('result', weightFile)
    if checkToSkip(weightFile, overwrite):
        sys.exit(0)
    makedirsforfile(weightFile)

    test()
    print '-'*70
    best_perf = -10
    best_alpha = None
    
    data = load_data(os.path.join('result', inputeFile))
    for i in range(1):
        perf, alpha = coordinate_ascent(data)
        if perf > best_perf:
            best_perf = perf
            best_alpha = alpha
        print '*'*70
    print 'optimized wights:', ' '.join(['%g'%x for x in best_alpha])
    print 'best tuned performance:', best_perf

    open(weightFile, 'w').write(' '.join(map(str,best_alpha)))
    print 'optimized wight parameters have written into %s' % weightFile



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] """)

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--inputeFile", default='qid.img.lable.feature.txt', type="string", help="file stored all score from different methods")
    parser.add_option("--weightFile", default='optimized_wights.txt', type="string", help="optimized wight will be written in the file")
    
    (options, args) = parser.parse_args(argv)
           
    return process(options)


if __name__ == "__main__":
    sys.exit(main())    




    
    
