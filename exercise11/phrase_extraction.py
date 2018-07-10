from alignment import Alignment
import numpy as np

def extract_phrase(sent_alignment, sent_pair):
    result = []
    e_length = len(sent_pair[0])
    f_length = len(sent_pair[1])

    for e_start in range(1,e_length):
        for e_end in range(e_start,e_length):
            f_start, f_end = f_length,0
            for e,f in zip(sent_alignment[0],sent_alignment[1]):
                if e_start<=e<=e_end:
                    f_start=min(f,f_start)
                    f_end = max(f,f_end)
            r = add_extract(f_start,f_end,e_start,e_end,sent_alignment, f_length)
            if r:
                e_phrase = " ".join(np.take(sent_pair[0],r[0]))
                f_phrase = " ".join(np.take(sent_pair[1],r[1]))
                result.append((e_phrase,f_phrase))
    with open('result','a') as rf:
        for phr in result:
            rf.write(phr[0])
            rf.write("\t")
            rf.write(phr[1])
            rf.write("\n")

def add_extract(f_start,f_end,e_start,e_end,sent_alignment, f_len):
    if f_end<0:
        return []
    for e, f in zip(sent_alignment[0], sent_alignment[1]):
        if ((f_start <= f <= f_end) and
                (e < e_start or e > e_end)):
            return []
    E = []
    f_s = f_start
    while True:

        f_e = f_end
        while True:
            E.append([i for i in range(e_start,e_end)])
            E.append([j for j in range(f_s,f_e)])
            f_e+=1
            if f_e in sent_alignment[1] or f_e > f_len:
                break
        f_s-=1
        if f_s in sent_alignment[1] or f_s < 0:
            break
    return E

def sent_to_num(sent):
    return [x for x in range(len(sent))]


def process(source, target, alignment):
    a = Alignment()
    a.read_alignment(alignment)
    with open(source) as sf, open(target) as tf:
        nr = 1
        for line1, line2 in zip(sf,tf):
            source_sent = line1.strip().split()
            target_sent = line2.strip().split()
            #print(source_sent)
            #print(target_sent)
            extract_phrase(a.alignment[nr],[source_sent,target_sent])
            nr+=1

if __name__=="__main__":
    process('source','target','alignment')