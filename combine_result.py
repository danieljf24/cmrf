import os,sys
from util.tools import readQidQuery
from basic.common import ROOT_PATH, printMessage, checkToSkip, makedirsforfile


def readAnnotations(inputfile):
    return [(str.split(x)[0], str.split(x)[1].strip()) for x in open(inputfile).readlines()]

def checkSameList(list1, list2):
    if len(list1) == len(list2):
        for i in xrange(len(list1)):
            if list1[i] != list2[i]:
                return False
    else:
        return False
    return True


def process(options, collection):
    rootpath = options.rootpath
    overwrite = options.overwrite
    inputfile = options.inputfile
    resultname = options.resultname

    result_file = os.path.join('result', resultname)
    if checkToSkip(result_file, overwrite):
       sys.exit(0)
    makedirsforfile(result_file)

    # inpute of query
    qid_query_file = os.path.join(rootpath, collection, 'Annotations', 'qid.text.txt')
    qid_list, query_list = readQidQuery(qid_query_file)

    num2file = {}
    num2file[0] = os.path.join(rootpath, collection, 'Annotations', 'Image', 'concepts%s.txt' % collection)
    method_count = 1
    for line in open(os.path.join('result',inputfile)).readlines():
        num2file[method_count] = line.strip()
        method_count +=1

    fout = open(result_file, "w")
    
    for qid in qid_list:
        name2feature = {}
        for fnum in xrange(method_count):
            data_file = os.path.join( num2file[fnum], '%s.txt' % qid)
            data = readAnnotations(data_file)
            data.sort(key=lambda v:v[0], reverse=True)
            names = [x[0] for x in data]
            labels = [x[1] for x in data]
            # print 'fnum %d' % fnum
            if fnum == 0:
                key_names = names
                for i in xrange(len(names)):
                    name2feature[names[i]] = [labels[i]]
            else:
                assert(checkSameList(key_names, names))
                for i in xrange(len(names)):
                    name2feature[names[i]].append(labels[i])
        for img in key_names:
            fout.write('%s ' % qid + img + ' ' + ' '.join(name2feature[img]) + '\n')  
    fout.close()

    print 'Combined result of different written into %s' % result_file


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--inputfile", default='individual_result_pathes.txt', type="string", help="path of different individul methods")
    parser.add_option("--resultname", default='qid.img.lable.feature.txt', type="str", help="result file name")

    
    
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())    
