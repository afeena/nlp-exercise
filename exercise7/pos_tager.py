from collections import defaultdict, Counter
import string
import cProfile
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
        self.n_grams_sums = [0 for i in range(self.n)]
        self.tags_sums = [0 for i in range(self.n)]


    def sent_to_n_grams(self, sent, n, destination):
        for i in range(n-1,len(sent)):
            if n==2:
                h = sent[i-1]
                w = sent[i]
                if h not in destination[n-1]: destination[n-1][h] = {}
                if w not in destination[n-1][h]: destination[n-1][h][w] = 0
                destination[n - 1][h][w] += 1
            if n==1:
                uni = sent[i]
                destination[n - 1][uni]+= 1

        if n>1:
            self.sent_to_n_grams(sent, n - 1, destination)

    def read_dataset(self,text_fname, tags_fname, size):
        counter = 0
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
                counter+=1
                if counter==size:
                    break

        del self.n_grams[0]["<s>"]


    def count(self):
        self.get_counts_of_counts(self.n_grams,self.n_grams_counts)
        self.get_counts_of_counts(self.tags,self.tags_counts)
        self.precalc_lambdas(5)
        self.n_grams_sums[0] = sum(self.n_grams[0].values())
        self.tags_sums[0] = sum(self.tags[0].values())

    def get_counts_of_counts(self, source, dest):
        for i,d in enumerate(source):
            for k,v in source[i].items():
                if type(v) is dict:
                    dest[i].update(v.values())
                else:
                    dest[i].update(source[i].values())
                    break

    def get_prob_word(self,word):
        if self.n_grams[0][word] > 0:
            return self.n_grams[0][word] / self.n_grams_sums[0]
        else:
            return 1 / len(self.n_grams[0])

    def get_prob_tag(self,tag):
        return self.tags[0][tag] / self.tags_sums[0]

    def get_prob_word_given_tag(self, word, tag):
        if self.tag_text[(word,tag)] >0:
            return self.tag_text[(word, tag)]/self.tags[0][tag]
        else:
            return (self.n_grams[0][word]+1)/(self.n_grams_sums[0]+len(self.n_grams[0])+1)


    def get_prob_tag_bigram(self):
        tags_voc = self.tags[0].keys()
        self.tags_bigrams_probs = defaultdict(int)

        histories = list(tags_voc)
        histories.append("<s>")
        #check the formula

        for t1 in tags_voc:
            for t2 in histories:
                if t2 not in self.tags[1] or t1 not in self.tags[1][t2]:
                    r = self.tags[0][t1]/self.tags_sums[0]
                else:
                    r = self.tags[1][t2][t1]/self.tags[0][t2]
                self.tags_bigrams_probs[(t2,t1)] = r

    def prob_bigram_tag(self,g1,g2):
        if self.tags[0][g2]==0:
            return self.tags[0][g1]/self.tags_sums[0]
        return self.tags[1][(g2,g1)]/self.tags[0][g2]

    def get_prob_tag_word(self,tag,word):
        #r = self.tag_text[(word,tag)]/self.n_grams[0][word]
        r = self.turing_good_discounting(tag,word)
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
        if type(self.n_grams[n-1][h]) is int:
            c= self.n_grams[n-1][h]
            if c >= 1 and c <= k:
                sum += (c / self.n_grams[0][h]) * self.lambdas[c - 1]
        else:
            for w,c in self.n_grams[n-1][h].items():
                if c>=1 and c<=k:
                    sum+=(c/self.n_grams[0][h])*self.lambdas[c-1]

        self.a_h_cache[h] = sum
        return sum


    def without_smoothing(self,word,tag1,tag2):
        if tag2==None:
            return 0
        try:
            a1 = self.tag_text[(word,tag1)]/self.tags[0][tag1]
            a2 = self.tags[1][tag2][tag1]/self.tags[0][tag2]
        except KeyError:
            return 0

        return a1*a2


    def turing_good_discounting(self, tag, word, n=2):
        #Katz model using turing-good discounting
        #assume k==5
        k=5
        r = self.tag_text[(word,tag)]

        if n==1:
            if self.tags[0][tag]>0:
                return self.tags[0][tag]/self.n_grams_sums[0]
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
    def __init__(self, verbose):
        self.ngram = NGram(2)
        self.verbose = verbose


    def train(self, text_train, tags_train):
        self.ngram.read_dataset(text_train, tags_train, size=None)
        self.ngram.count()
        self.ngram.get_prob_tag_bigram()
        self.tags_voc = list(self.ngram.tags[0].keys())


    def test(self, text_file, tag_file):
        references = []
        results = []

        with open(text_file) as text, open(tag_file) as tag:
            for text_line, tag_line in zip(text,tag):
                references.append([t for t in tag_line.split()])
                to_tag = [s.lower() for s in text_line.split()]
                #to_tag.insert(0,"<s>")
                #to_tag.append("<e>")
                res = self.tag_sequence(to_tag)
                results.append(res)


        if self.verbose:
            self.score_by_tokens(references,results)

    def score_by_tokens(self, ref, res):
        pos_tags = list(self.ngram.tags[0].keys())
        post_tags_wrong = defaultdict(int)
        post_tags_occ = defaultdict(int)
        for rf, rs in zip(ref, res):
            for r1, r2 in zip(rf, rs):
                if r1 != r2:
                    post_tags_wrong[r1]+=1
                post_tags_occ[r1]+=1


        print("error rate by tokens", sum(post_tags_wrong.values())/sum(post_tags_occ.values()))

        for k,v in post_tags_wrong.items():
            print("{}:{}\n".format(k,v/post_tags_occ[k]))


    def tag_sequence(self, sequence):
        if type(sequence) is str:
            sequence = [s.lower() for s in sequence.split()]
        g0 = "<s>"

        backpointers = []
        prev_tag = g0
        current_tag = None
        for w in sequence:
            max = 0
            for tag in self.tags_voc:
                r = ((self.ngram.get_prob_word(w)*self.ngram.get_prob_tag_word(tag,w))/self.ngram.get_prob_tag(tag))*self.ngram.tags_bigrams_probs[prev_tag,tag]
                #add-one
                #r = self.ngram.get_prob_word_given_tag(w,tag) * self.ngram.tags_bigrams_probs[prev_tag,tag]

                #without smoothing
                #r = self.ngram.without_smoothing(w,tag,prev_tag)
                if r>max:
                    max = r
                    current_tag = tag
            backpointers.append(current_tag)
            prev_tag = current_tag

        #print(backpointers)
        return backpointers



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_text', default="./wsj/wsj.text.tr")
    parser.add_argument('-train_pos', default="./wsj/wsj.pos.tr")
    parser.add_argument('-test_text', default="./wsj/wsj.text.test")
    parser.add_argument('-test_pos',default="./wsj/wsj.pos.test")
    parser.add_argument('--verbose', help="show error rates", default=None)
    parser.add_argument('--interactive', help="console mode tagging", default=None)

    args = parser.parse_args()
    pt = POS_tagger(args.verbose)
    pt.train(args.train_text,args.train_pos)
    print("train_done")
    pt.test(args.test_text,args.test_pos)
    if args.interactive:
        print("prinv Ctrl+C to exit")
        while True:
            seq = input()
            print(pt.tag_sequence(seq))



