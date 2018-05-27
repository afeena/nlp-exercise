from collections import defaultdict, Counter
import string
import numpy as np

import argparse


class NGram():
    def __init__(self, n):
        self.n = n
        self.n_grams = [defaultdict(int) for i in range(self.n)]
        self.n_grams_counts = [Counter() for i in range(self.n)]


    def sent_to_n_grams(self, sent,n):
        # self.n_grams[0][sent[0]]+=1
        # for i in range(1,len(sent)):
        #     big = (sent[i-1],sent[i])
        #     bigrams[big]+=1
        #     unigrams[sent[i]]+=1

        #sent.insert(0,",<s>")
        for i in range(n-1,len(sent)):
            self.n_grams[n-1][" ".join(sent[i-(n-1):i+1])]+=1
        if n>1:
            self.sent_to_n_grams(sent,n-1)

    def read_dataset(self,f_name):
        with open(f_name) as ep:
            for line in ep:
                sent = line.split()
                sent = [s.lower() for s in sent if s not in string.punctuation]
                #sent.append("<e>")
                if len(sent)==0:
                    continue
                self.sent_to_n_grams(sent, self.n)
        #del self.n_grams[0]["<e>"]

    def get_counts_of_counts(self):
        for i,d in enumerate(self.n_grams):
            self.n_grams_counts[i].update(self.n_grams[i].values())


    def calculate_lamba_r(self,r,k, n):
        n_1 = self.n_grams_counts[n-1][1]
        return (1 - (((r + 1) * self.n_grams_counts[n-1][r + 1]) / r * self.n_grams_counts[n-1][r])) / (1 - ((k + 1) * self.n_grams_counts[n-1][k + 1] / n_1))

    def calculate_a_h(self,n,h,k):
        #p.215
        lambdas = []
        sum = 0
        for i in range(k+1):
            lambdas.append(self.calculate_lamba_r(i,k,n))
        for k,v in self.n_grams[n-1].items():
            if v>=1 and v<=k:
                sum+=(v/self.n_grams[n-1][h])*lambdas[v-1]

        return sum

    def turing_good_discounting(self, h,w, n=2):
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

        r = self.n_grams[1][" ".join([h,w])]
        if r>0:
            if r<k:
                lambda_r = self.calculate_lamba_r(r,k,self.n)
                r_star = (1-lambda_r)*r
            else:
                r_star = r
            p = r_star / self.n_grams[n-2][h]
        else:
            #P.220
            p = self.n_grams[0][h]/sum(self.n_grams[0].values())
        return p

    def get_counts_big(h):
        s = 0
        for k,v in bigrams.items():
            if k[0]==h:
                s+=v

        return s

    def absolute_discounting(self, w,h, n):
        #p.179
        #assume history independent h
        b = self.n_grams_counts[n-1][1]/(self.n_grams_counts[n-1][1]+2*self.n_grams_counts[n-1][2])
        if self.n_grams[n-1][" ".join([h,w])]>0:
            p = (self.n_grams[n-1][" ".join([h,w])]-b)/self.n_grams[n-2][h]
        else:
            nom = (sum(self.n_grams[n-2].values())-self.n_grams[n-2][h])
            beta = self.n_grams[n-2][w]/sum(self.n_grams[n-2].values())
            p = b*(nom/self.n_grams[n-2][h])*beta
        return p


    def interpolated_absolute_discounting(self, w, h,n):
        # p.192
        # assume history independent b
        b = self.n_grams_counts[n-1][1]/(self.n_grams_counts[n-1][1]+2*self.n_grams_counts[n-1][2])
        nom = (sum(self.n_grams[n-2].values())-self.n_grams[n-2][h])
        beta = self.n_grams[n-2][w]/sum(self.n_grams[n-2].values())
        p = (max((self.n_grams[n-1][" ".join([h, w])] - b),0) / self.n_grams[n-2][h]) + b * (nom / self.n_grams[n-2][h]) * beta
        return p

class Generator:
    def generate(self, l, d):
        sentence = []
        word_list = list(d.n_grams[0].keys())
        first_word_index = np.random.choice([i for i in range(len(word_list))])
        first_word = word_list.pop(first_word_index)
        sentence.append(first_word)
        current_word = first_word
        for i in range(l-1):
            max = 0
            next_w_i = -1
            for i,w in enumerate(word_list):
                p = d.interpolated_absolute_discounting(current_word,w,2)
                if p>max:
                    max = p
                    next_w_i = i

            sentence.append(word_list.pop(next_w_i))
        print(sentence)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./Europarl.txt')
    parser.add_argument('-n', type=int, default=2)

    args = parser.parse_args()

    d = NGram(args.n)
    d.read_dataset("test")
    d.get_counts_of_counts()

    # print(len(bigrams))
    #
    # #get counts of counts
    # counts_of_counts_bi = Counter(bigrams.values())
    # counts_of_counts_uni = Counter(unigrams.values())
    p1 = d.absolute_discounting("people", "the",2)
    p2 = d.turing_good_discounting("people", "the")
    p3 = d.interpolated_absolute_discounting("people","the",2)
    print(p1,p2,p3)
    gen  =  Generator()
    gen.generate(5,d)

