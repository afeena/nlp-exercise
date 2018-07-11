from collections import defaultdict
class Alignment:
    def __init__(self):
        self.alignment = defaultdict(list)


    def read_alignment(self,filename):
        with open(filename) as f:
            alignment_data = f.read().split("\n\n")
        for i,sent in enumerate(alignment_data):
            sent_tok = sent.split("\n")
            sent_nr = int(sent_tok[0].split()[1])
            sources = []
            targets = []
            try:
                for s in sent_tok[1:]:
                    sources.append(int(s.split()[1]))
                    targets.append(int(s.split()[2]))
            except IndexError:
                pass

            self.alignment[sent_nr].append(sources)
            self.alignment[sent_nr].append(targets)