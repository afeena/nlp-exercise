from collections import defaultdict, Counter
import string
import numpy as np

bigrams = defaultdict(int)
unigrams = defaultdict(int)

def line_to_big(l):
    unigrams[l[0]]+=1
    for i in range(1,len(l)):
        big = (l[i-1],l[i])
        bigrams[big]+=1
        unigrams[l[i]]+=1


def read_dataset():
    with open("test") as ep:
        for line in ep:
            sent = line.split()
            sent = [s.lower() for s in sent if s not in string.punctuation]
            if len(sent)==0:
                continue
            line_to_big(sent)


def turing_good_discounting(h,w,c_bi, c_u):
    # it=1
    # discounts = {}
    # for count in c_bi.items():
    #     try:
    #         r_star = (it+1)*c_bi[it]/count[1]
    #         discounts[it] = r_star
    #         it+=1
    #     except IndexError:
    #         r_star = 0
    #         discounts[it] = r_star
    k=5
    n_1 = c_bi[1]
    r = bigrams[(h,w)]
    if r>0:
        if r<k:
            lambda_r = (1-(((r+1)*c_bi[r+1])/r*c_bi[r]))/(1-((k+1)*c_bi[k+1]/n_1))
            r_star = (1-lambda_r)*r
        else:
            r_star = r
        p = r_star / get_counts_big(h)
    else:
        #P.220
        p = unigrams[h]/sum(unigrams.values())
    return p

def get_counts_big(h):
    s = 0
    for k,v in bigrams.items():
        if k[0]==h:
            s+=v

    return s

def absolute_discounting(w,h,c_bi, c_u):
    #p.179
    #assume history independent h
    b = c_bi[1]/(c_bi[1]+2*c_bi[2])
    if bigrams[(h,w)]>0:
        p = (bigrams[(h,w)]-b)/get_counts_big(h)
    else:
        nom = (sum(unigrams.values())-unigrams[h])
        beta = unigrams[w]/sum(bigrams.values())
        p = b*(nom/get_counts_big(h))*beta
    return p


def interpolated_absolute_discounting(w,h,c_bi, c_u):
    # p.192
    # assume history independent b
    b = c_bi[1] / (c_bi[1] + 2 * c_bi[2])
    nom = (sum(unigrams.values())-unigrams[h])
    beta = unigrams[w] / sum(bigrams.values())
    p = (max((bigrams[(h, w)] - b),0) / get_counts_big(h)) + b * (nom / get_counts_big(h)) * beta
    return p

def generate(l, c_bi, c_u):
    sentence = []
    word_list = list(unigrams.keys())
    first_word_index = np.random.choice([i for i in range(len(word_list))])
    first_word = word_list.pop(first_word_index)
    sentence.append(first_word)
    current_word = first_word
    for i in range(l-1):
        max = 0
        next_w_i = -1
        for i,w in enumerate(word_list):
            p = turing_good_discounting(current_word,w, c_bi, c_u)
            if p>max:
                max = p
                next_w_i = i

        sentence.append(word_list.pop(next_w_i))
    print(sentence)

if __name__=="__main__":
    read_dataset()
    print(len(bigrams))

    #get counts of counts
    counts_of_counts_bi = Counter(bigrams.values())
    counts_of_counts_uni = Counter(unigrams.values())
    p1 = absolute_discounting("people", "the",counts_of_counts_bi, counts_of_counts_uni)
    p2 = turing_good_discounting("people", "the", counts_of_counts_bi, counts_of_counts_uni)
    p3 = interpolated_absolute_discounting("people","the",counts_of_counts_bi, counts_of_counts_uni)
    print(p1,p2,p3)
    generate(5,counts_of_counts_bi, counts_of_counts_uni)

