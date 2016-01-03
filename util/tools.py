import os


def readQidQuery(inputfile):
    data = [ x.strip().split(" ", 1) for x in open(inputfile).readlines()]
    qid = [x[0] for x in data]
    query = [x[1] for x in data]
    return (qid, query)



def readNnData(inputfile, n_top = 50):    # format: key data1 data2 data1 data2
    data = [ x.strip().split(' ', 1) for x in open(inputfile).readlines()]
    key = [x[0] for x in data]
    data_list = [x[1] if len(x)>1 else '' for x in data]

    keyid2data = {}
    for index in range(len(key)):
        data1_data2 = data_list[index].split()
        length = 2*n_top if len(data1_data2) > 2*n_top else len(data1_data2)
        data1 = [data1_data2[x] for x in range(0,length,2)]
        data2 = [float(data1_data2[x]) for x in range(1,length,2)]
        assert(len(data1) == len(data2)),"data1: %d, data2:%d" % (len(data1),len(data2))
        keyid2data[key[index]] = zip(data1, data2)
    return keyid2data
    # return (key, keyid2data)


def readQueryClickture(inputfile):    # format: key \t data1 data2 data1 data2
    data = [ x.strip().split('\t') for x in open(inputfile).readlines()]
    key = [x[0] for x in data]
    data_list = [x[1] if len(x)>1 else '' for x in data]

    keyid2data = {}
    for index in range(len(key)):
        data1_data2 = data_list[index].split()
        length = len(data1_data2)
        data1 = [data1_data2[x] for x in range(0,length,2)]
        data2 = [data1_data2[x] for x in range(1,length,2)]
        assert(len(data1) == len(data2)),"data1: %d, data2:%d" % (len(data1),len(data2))
        keyid2data[index] = zip(data1, data2)
    return keyid2data
    # return (key, keyid2data)



def readImageClickture(inputfile, n_top_query):   # format: img_id /t query /t click ...
    data = [ x.strip().split('\t', 1) for x in open(inputfile).readlines()]
    img = [x[0] for x in data]
    query_click = [x[1] for x in data]

    img2query_clc = {}
    for index in range(len(img)):
        querylist = query_click[index].split('\t')
        length = 2*n_top_query if len(querylist) > 2*n_top_query else len(querylist)
        query = [querylist[x] for x in range(0,length,2)]
        click = [int(querylist[x]) for x in range(1,length,2)]
        assert(len(query) == len(click))
        img2query_clc[img[index]] = zip(query, click)

    return img2query_clc


def writeRankingResult(outputfileDir, qid2iid_label_score):
    try:
        os.makedirs(outputfileDir)
    except Exception, e:
        #print e
        pass
    for qid in qid2iid_label_score:
        fout = open(os.path.join(outputfileDir, qid+'.txt'), "w")
        fout.write("".join(["%s %g\n" % (iid,score) for (iid,lab,score) in qid2iid_label_score[qid]]))
        fout.close()
    

def writeDCGResult(outputfileDir, qid2dcg):
    try:
        os.makedirs(outputfileDir)
    except Exception, e:
        #print e
        pass
    fout = open(os.path.join(outputfileDir, 'DCG@25.txt'), "w")
    overall_DCG = sum(qid2dcg.values())/len(qid2dcg.values())
    fout.write("Overall: %g\n" % overall_DCG )
    fout.write("\n".join(["%s %g" % (k,v) for (k,v) in qid2dcg.iteritems() ]))
    fout.close()
