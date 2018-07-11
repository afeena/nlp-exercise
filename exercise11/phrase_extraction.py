from alignment import Alignment
import numpy as np
import argparse

def extract_phrase(sent_alignment, sent_pair, output):
    result = []
    e_length = len(sent_pair[0])
    f_length = len(sent_pair[1])

    for e_start in range(e_length):
        for e_end in range(e_start,e_length):
            f_start, f_end = f_length-1,-1
            for e,f in zip(sent_alignment[0],sent_alignment[1]):
                if e_start<=e<=e_end:
                    f_start=min(f,f_start)
                    f_end = max(f,f_end)
            r = add_extract(f_start,f_end,e_start,e_end,sent_alignment, f_length)
            if r:
                e_phrase = " ".join(np.take(sent_pair[0],r[0]))
                f_phrase = " ".join(np.take(sent_pair[1],r[1]))
                result.append((e_phrase,f_phrase))

    with open(output,'a') as rf:
        for phr in result:
            rf.write("{}\t - \t{}\n".format(phr[0],phr[1]))
        rf.write("\n")

def add_extract(f_start,f_end,e_start,e_end,sent_alignment, f_len):
    if f_end<0:
        return []
    for e, f in zip(sent_alignment[0], sent_alignment[1]):
        if ((f_start <= f <= f_end) and
                (e < e_start or e > e_end)):
            return []
    phrase_set = []
    f_s = f_start
    while True:

        f_e = f_end
        while True:
            phrase_set.append([i for i in range(e_start,e_end+1)])
            phrase_set.append([j for j in range(f_s,f_e+1)])
            f_e+=1
            if f_e in sent_alignment[1] or f_e == f_len:
                break
        f_s-=1
        if f_s in sent_alignment[1] or f_s < 0:
            break
    return phrase_set

def sent_to_num(sent):
    return [x for x in range(len(sent))]


def process(source, target, alignment, output):
    a = Alignment()
    a.read_alignment(alignment)
    with open(source) as sf, open(target) as tf:
        nr = 1
        for line1, line2 in zip(sf,tf):
            source_sent = line1.strip().split()
            target_sent = line2.strip().split()
            #print(source_sent)
            #print(target_sent)
            extract_phrase(a.alignment[nr],[source_sent,target_sent], output)
            nr+=1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="source file name", default="source")
    parser.add_argument("-t", "--target", help="target file name", default="target")
    parser.add_argument("-a", "--alignment", help="alignment file name", default="alignment")
    parser.add_argument("-o", "--output", help="output file name", default="phrase_results.txt")
    args = parser.parse_args()

    process(args.source, args.target, args.alignment, args.output)