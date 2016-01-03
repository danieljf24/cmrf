import os
import sys
import numpy as np
from basic.common import checkToSkip
    

def process(options):

    overwrite = options.overwrite
    inputeFile = options.inputeFile
    weightFile = options.weightFile
    resultFile = options.resultFile

        
    weightFile = os.path.join('result', weightFile)
    weight = open(weightFile).readline().strip().split()
    weight = np.array(weight, dtype=np.float)

    resultFile = os.path.join('result', resultFile)
    if checkToSkip(resultFile, overwrite):
        sys.exit(0)
    fout = open(resultFile, 'w')

    done = 0
    for line in open(os.path.join('result', inputeFile)):
        elems = line.strip().split()
        vecs = map(float, elems[3:])
        vecs = np.array(vecs, dtype=np.float)
        assert(len(weight) == len(vecs))

        fout.write(" ".join(elems[:2]) + " " + str(np.dot(weight, vecs)) + '\n')

        done += 1
        if done % 10000 == 0:
            print done, 'Done'

    fout.close()
    print "final score result after relevance fusion have written in %s" % resultFile



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] """)

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--inputeFile", default='qid.img.lable.feature.txt', type="string", help="file stored all score from different methods")
    parser.add_option("--weightFile", default='optimized_wights.txt', type="string", help="optimized wight will be written in the file")
    parser.add_option("--resultFile", default='fianl.result.txt', type="string", help="final score after relevance fusion")
    
    (options, args) = parser.parse_args(argv)
           
    return process(options)


if __name__ == "__main__":
    sys.exit(main())    




    
    
