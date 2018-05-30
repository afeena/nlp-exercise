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
        for i in range(n-1,len(sent)):
            self.n_grams[n-1][" ".join(sent[i-(n-1):i+1])]+=1
        if n>1:
            self.sent_to_n_grams(sent,n-1)

    def read_dataset(self,f_name):
        with open(f_name) as ep:
            for line in ep:
                sent = line.split()
                sent = [s.lower() for s in sent if s not in string.punctuation]
                sent.insert(0,"<s>")
                sent.append("<e>")
                if len(sent)==0:
                    continue
                self.sent_to_n_grams(sent, self.n)
        del self.n_grams[0]["<e>"]
        del self.n_grams[0]["<s>"]

    def get_counts_of_counts(self):
        for i,d in enumerate(self.n_grams):
            self.n_grams_counts[i].update(self.n_grams[i].values())


    def calculate_lamba_r(self,r,k, n):
        #P.216
        n_1 = self.n_grams_counts[n-1][1]
        nom =  (1 - (((r + 1) * self.n_grams_counts[n-1][r + 1])/(r * self.n_grams_counts[n-1][r])))
        den = (1 - ((k + 1) * self.n_grams_counts[n-1][k + 1] / n_1))
        return nom/den

    def calculate_a_h(self,n,h,k):
        #p.215
        lambdas = []
        sum = 0
        for i in range(1,k+1):
            lambdas.append(self.calculate_lamba_r(i,k,n))
        for w,v in self.n_grams[n-1].items():
            if w.split()[0]==h and v>=1 and v<=k:
                sum+=(v/self.n_grams[0][h])*lambdas[v-1]

        return sum

    def turing_good_discounting(self, w, h, n):
        #Katz model using turing-good discounting
        #assume k==5
        k=5
        r = self.n_grams[n-1][" ".join([h,w])]

        if n==1:
            if self.n_grams[0][w]>0:
                return self.n_grams[0][w]/sum(self.n_grams[0].values())
            else:
                return 1/len(self.n_grams[0])

        if r>0:
            if r<k:
                lambda_r = self.calculate_lamba_r(r,k,n)
                r_star = (1-lambda_r)*r
            else:
                r_star = r
            p  = r_star / self.n_grams[n - 2][h]
        else:
            #P.220
            #backoff according Katz model
            p = self.calculate_a_h(n,h,k)*self.turing_good_discounting(w, " ".join(h.split()[1:]),n-1)
        return p


    def absolute_discounting(self, w, h, n):
        #p.179
        #assume history independent b

        if n == 1:
            if self.n_grams[0][w] > 0:
                return self.n_grams[0][w] / sum(self.n_grams[0].values())
            else:
                return 1 / len(self.n_grams[0])

        #p.180
        #calculate discounting parameter
        b = self.n_grams_counts[1][1]/(self.n_grams_counts[1][1]+2*self.n_grams_counts[1][2])


        if self.n_grams[n-1][" ".join([h,w])]>0:
            #calculate probability of seen n-gram
            p = (self.n_grams[n-1][" ".join([h,w])]-b)/self.n_grams[n-2][h]
        else:
            #backoff
            nom = len(self.n_grams[0])-(len(self.n_grams[0])-self.n_grams[0][h])
            p = b*(nom/self.n_grams[n-2][h])*self.absolute_discounting(w," ".join(h.split()[1:]),n-1)
        return p


    def clac_n_plus(self, h,w,n):
        s = 0
        for w in self.n_grams[0].keys():
            for k,v in self.n_grams[n-1].items():
                if k == " ".join([h,w]) and v>=0:
                    s+=1
        return s

    def interpolated_absolute_discounting(self, w, h,n):
        # p.192
        # assume history independent b

        #if n==1 just take the relative frequence
        if n==1:
            if self.n_grams[0][w]>0:
                return self.n_grams[0][w]/sum(self.n_grams[0].values())
            else:
                return 1/len(self.n_grams[0])

        b = self.n_grams_counts[n-1][1]/(self.n_grams_counts[n-1][1]+2*self.n_grams_counts[n-1][2])
        nom = len(self.n_grams[0])-(len(self.n_grams[0])-self.n_grams[0][h])

        p = (max((self.n_grams[n-1][" ".join([h, w])] - b),0) / self.n_grams[n-2][h]) + \
            (b * (nom / self.n_grams[n-2][h])) * self.interpolated_absolute_discounting(w," ".join(h.split()[1:]),n-1)
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
                p = d.turing_good_discounting(w,current_word,2)
                if p>max:
                    max = p
                    next_w_i = i

            sentence.append(word_list.pop(next_w_i))
            current_word = sentence[-1]
        print(sentence)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./Europarl.txt')
    parser.add_argument('-n', type=int, default=2)
    parser.add_argument('-sl', type=int, default=5, help="sentence length")

    args = parser.parse_args()

    d = NGram(args.n)
    d.read_dataset(args.data)
    d.get_counts_of_counts()

    #testing part
    p1= d.turing_good_discounting("people","the",2)
    p2 = d.turing_good_discounting("people","table",2)
    p3= d.absolute_discounting("people","the",2)
    p4 = d.absolute_discounting("people","table",2)
    p5= d.interpolated_absolute_discounting("people","the",2)
    p6 = d.interpolated_absolute_discounting("people","table",2)
    print(p1, p2, p3,p4,p5,p6)

    #generator. can be too slow
    gen = Generator()
    gen.generate(args.sl,d)


