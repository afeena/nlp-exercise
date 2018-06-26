import numpy
import argparse
from collections import Counter


def word_error_rate(hyp, ref):


    hyp_len = len(hyp)
    ref_len = len(ref)
    if ref_len==0 and hyp_len==0:
        return 0
    distances = numpy.zeros((ref_len+1,hyp_len+1))
    for i in range(ref_len+1):
        for j in range(hyp_len+1):
            if i == 0:
                distances[0][j] = j
            elif j == 0:
                distances[i][0] = i

    for i in range(1,ref_len+1):
        for j in range(1,hyp_len+1):
            if hyp[j-1]==ref[i-1]:
                distances[i,j]=distances[i-1][j-1]
            else:
                distances[i][j] = min(distances[i - 1][j-1], distances[i][j-1], distances[i - 1][j])+1


    edit_dist = distances[ref_len][hyp_len]
    return float(edit_dist)/float(ref_len)

def per(hyp,ref):
    hyp_len = len(hyp)
    ref_len = len(ref)
    if ref_len==0 and hyp_len==0:
        return 0.0
    ref_counter = Counter(ref)
    hyp_counter = Counter(hyp)

    s = 0
    for w in hyp:
        s+=abs(ref_counter[w]-hyp_counter[w])

    d_per = s#0.5*(abs(ref_len-hyp_len)+s)


    return d_per/ref_len




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ref", help="refrence file")
    parser.add_argument("-hyp", help="hypothesis file")

    args = parser.parse_args()

    with open(args.ref) as rf:
        references = rf.read()


    with open(args.hyp) as rf:
        hypothesis = rf.read()

    hypothesis = [h.split() for h in hypothesis.split("\n") if h!=""]
    references = [r.split() for r in references.split("\n") if r!=""]
    wers = []
    pers = []
    for h,r in zip(hypothesis,references):
        e = word_error_rate(h,r)
        p = per(h,r)
        pers.append(p)
        wers.append(e)
        print(e,p)

    print(sum(wers)/len(wers))
    print(sum(pers)/len(pers))