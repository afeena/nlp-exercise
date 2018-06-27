from collections import defaultdict

cent_sent = [
"ok-voon ororok sprok .",
"ok-drubel ok-voon anok plok sprok .",
"erok sprok izok hihok ghirok .",
"ok-voon anok drok brok jok .",
"wiwok farok izok stok .",
"lalok sprok izok jok stok .",
"lalok farok ororok lalok sprok izok enemok .",
"lalok brok anok plok nok .",
"wiwok nok izok kantok ok-yurp .",
"lalok mok nok yorok ghirok clok .",
"lalok nok crrrok hihok yorok zanzanok .",
"lalok rarok nok izok hihok mok ."
]

arc_sent = [
"at-voon bichat dat .",
"at-drubel at-voon pippat rrat dat .",
"totat dat arrat vat hilat .",
"at-voon krat pippat sat lat .",
"totat jjat quat cat .",
"wat dat krat quat cat .",
"wat jjat bichat wat dat vat eneat .",
"iat lat pippat rrat nnat .",
"totat nnat quat oloat at-yurp .",
"wat nnat gat mat bat hilat .",
"wat nnat arrat mat zanzanat .",
"wat nnat forat arrat vat gat ."
]

arc_vocab = defaultdict(int)
cent_vocab = defaultdict(int)
arc_transl = defaultdict(dict)
for s1, s2 in zip(arc_sent, cent_sent):
    for word1, word2 in zip(s1.split(),s2.split()):
        arc_vocab[word1]+=1
        cent_vocab[word2]+=1
        if word2 not in  arc_transl[word1]:
            arc_transl[word1][word2]=1
        else:
            arc_transl[word1][word2] += 1


print(arc_vocab.keys())
print(cent_vocab.keys())


for k,v in arc_transl.items():
    for k1, v1 in v.items():
        if len(v)==1 and v1>1:
            print(k,k1,v1)