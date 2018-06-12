import numpy as np
from collections import defaultdict, Counter
import string
import numpy as np

import argparse


class NGram():
    def __init__(self, n):
        self.n = n
        self.n_grams = [defaultdict(int) for i in range(self.n)]
        self.n_grams_counts = [Counter() for i in range(self.n)]
        self.tags = [defaultdict(int) for i in range(self.n)]
        self.tags_counts = [Counter() for i in range(self.n)]
        self.tag_text = defaultdict(int)
        self.lambdas = []
        self.a_h_cache = {}


    def sent_to_n_grams(self, sent, n, destination):
        for i in range(n-1,len(sent)):
            if n==2:
                big = (sent[i-1],sent[i])
                destination[1][big] += 1
            if n==1:
                uni = sent[i]
                destination[0][uni]+= 1

        if n>1:
            self.sent_to_n_grams(sent, n - 1, destination)

    def read_dataset(self,text_fname, tags_fname):
        with open(text_fname) as text, open(tags_fname) as tag:
            for text_line, tag_line in zip(text,tag):
                sent = text_line.split()
                sent = [s.lower() for s in sent]
                if len(sent)==0:
                    continue
                sent.insert(0,"<s>")

                tag_sent = tag_line.split()
                tag_sent = [s for s in tag_sent]
                if len(tag_sent) == 0:
                    continue
                tag_sent.insert(0, "<s>")

                self.sent_to_n_grams(tag_sent, self.n, self.tags)
                self.sent_to_n_grams(sent, self.n,self.n_grams)

                for w,g in zip(sent,tag_sent):
                    self.tag_text[(w,g)]+=1


        del self.n_grams[0]["<s>"]
        #del self.tags[0]["<s>"]
        #self.tags[0]["<s>"]=1

    def count(self):
        self.uni_counts = sum(self.n_grams[0].values())
        self.uni_counts_tags = sum(self.tags[0].values())
        self.get_counts_of_counts(self.n_grams,self.n_grams_counts)
        self.get_counts_of_counts(self.tags,self.tags_counts)
        self.precalc_lambdas(5)
        self.count_prob_tag_bigram()



    def get_counts_of_counts(self, source, dest):
        for i,d in enumerate(source):
            dest[i].update(source[i].values())

    def get_prob_word(self,word):
        # if self.n_grams[0][word] > 0:
        #     return self.n_grams[0][word] / self.uni_counts
        # else:
        #     return 1 / len(self.n_grams[0])
        return self.n_grams[0][word]/self.uni_counts

    def get_prob_tag(self,tag):
        return self.tags[0][tag] / self.uni_counts_tags

    def count_prob_tag_bigram(self):

        tags_voc = self.tags[0].keys()
        self.tags_bigrams_probs = defaultdict(int)

        histories = list(tags_voc)
        #histories.append("<s>")

        #check the formula

        for t1 in tags_voc:
            if t1=="<s>":
                continue
            for t2 in histories:
                if self.tags[1][(t2,t1)]>0:
                    r = self.tags[1][(t2,t1)]/self.tags[0][t2]
                else:
                    r = self.tags[0][t1]/self.uni_counts_tags
                self.tags_bigrams_probs[(t2,t1)] = r

    def get_prob_tag_word(self,tag,word):
        r = self.tag_text[(word,tag)]/self.tags[0][tag]

        #r = self.turing_good_discounting(tag,word,2)
        return r

    def calculate_lamba_r(self,r,k, n):
        #P.216
        n_1 = self.tags_counts[n-1][1]
        nom =  (1 - (((r + 1) * self.tags_counts[n-1][r + 1])/(r * self.tags_counts[n-1][r])))
        den = (1 - ((k + 1) * self.tags_counts[n-1][k + 1] / n_1))
        return nom/den

    def precalc_lambdas(self,k):
        for i in range(1,k+1):
            self.lambdas.append(self.calculate_lamba_r(i,k,2))

    def calculate_a_h(self,n,h,k):
        #p.215
        if h in self.a_h_cache.keys():
            return self.a_h_cache[h]

        sum = 0
        for w,v in self.n_grams[n-1].items():
            if w[0]==h and v>=1 and v<=k:
                sum+=(v/self.n_grams[0][h])*self.lambdas[v-1]

        self.a_h_cache[h] = sum
        return sum


    def prob_bigram_tag(self,g1,g2):
        if self.tags[0][g2]==0:
            return 0.000001
        return self.tags[1][(g2,g1)]/self.tags[0][g2]

    def turing_good_discounting(self, tag, word, n=2):
        #Katz model using turing-good discounting
        #assume k==5
        k=5
        r = self.tag_text[(word,tag)]

        if n==1:
            if self.tags[0][tag]>0:
                return self.tags[0][tag]/self.uni_counts_tags
            else:
                return 1/len(self.tags[0])

        if r>0:
            if r<k:
                lambda_r = self.calculate_lamba_r(r,k,n)
                r_star = (1-lambda_r)*r
            else:
                r_star = r
            p  = r_star / self.n_grams[n - 2][word]
        else:
            #P.220
            #backoff according Katz model
            p = self.calculate_a_h(n,word,k)*self.turing_good_discounting(tag, "",1)
        return p


class POS_tagger:
    def __init__(self):
        self.ngram = NGram(2)
        self.cache = {}
        self.tag_vocab = []

    def train(self):
        self.ngram.read_dataset("./wsj/wsj.text.tr", "./wsj/wsj.pos.tr")
        self.ngram.count()
        self.tag_voc = list(self.ngram.tags[0].keys())

    def test(self, text_file, tag_file):
        references = []
        results = []
        err=0
        nt = 0
        with open(text_file) as text, open(tag_file) as tag:
            for text_line, tag_line in zip(text,tag):
                ref = [t for t in tag_line.split()]
                to_tag = [s.lower() for s in text_line.split()]
                #to_tag.insert(0,"<s>")
                #to_tag.append("<e>")
                res = self.tag_sequence(to_tag)
                for r1, r2 in zip(ref, res):
                    if r1 != r2:
                        err += 1
                    nt += 1
                #print(err/nt)
                #results.append(res)



        err_rate = err/nt
        print(err_rate)


    def tag_sequence(self, sequence):
        if type(sequence) is str:
            sequence = [s.lower() for s in sequence.split() if s not in string.punctuation]
        g0 = "<s>"
        backpointers = []
        prev_tag = g0
        current_tag = None
        q_prev = 1
        q_current = None
        for w in sequence:
            # if w in self.cache.keys():
            #     current_tag = self.cache[w]
            #     backpointers.append(current_tag)
            #     prev_tag = current_tag
            #     continue

            max = 0

            for tag in self.tag_voc:
                #r = ((self.ngram.get_prob_word(w)*self.ngram.get_prob_tag_word(tag,w))/self.ngram.get_prob_tag(tag))*self.ngram.tags_bigrams_probs[prev_tag,tag]
                r=self.ngram.get_prob_tag_word(tag,w)*self.ngram.prob_bigram_tag(tag,prev_tag)
                #r = self.ngram.get_prob(w,tag)*self.ngram.tags_bigrams_probs[prev_tag,tag]
                if r>max:
                    max = r
                    current_tag = tag

            backpointers.append(current_tag)
            #self.cache[w] = current_tag
            prev_tag = current_tag

        #print(backpointers)
        return backpointers



if __name__=="__main__":
    pt = POS_tagger()
    pt.train()
    print("train_done")
    seq = pt.tag_sequence("I am the student")
    print(seq)
    pt.test("./wsj/wsj.text.test", "./wsj/wsj.pos.test")




