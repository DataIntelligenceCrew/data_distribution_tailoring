import operator
import json
import random
from datetime import datetime

class BinaryDataset:
    def __init__(self, i, tuples, G1, G2, cost):
        self.id = i
        self.tuples = tuples
        self.N = len(tuples)
        self.N1 = len([i for i, v in enumerate(tuples) if v[1] == G1])
        self.N2 = len([i for i, v in enumerate(tuples) if v[1] == G2])
        self.P1 = float(self.N1)/self.N
        self.P2 = float(self.N2)/self.N
        self.G1, self.G2 = G1, G2
        self.O1, self.O2 = 0, 0
        self.C = cost
        self.seen = dict()

    
    def sample(self):
        random.seed(datetime.now())
        s = Sample(self.tuples[random.randint(0,self.N-1)], self.id, self.C)
        if s.rec[0] in self.seen:
            return None
        self.seen[s.rec[0]] = True
        return s 


    def update_with_sample(self, s):
        if s.rec[1] == self.G1:
            self.O1 += 1
            self.P1 = float(self.N1 - self.O1)/self.N
        else:
            self.O2 += 1
            self.P2 = float(self.N2 - self.O2)/self.N


class BinaryTarget:
    tuples = []

    def __init__(self, Q1=0, Q2=0, G1=0, G2=1):
        self.Q1 = Q1
        self.Q2 = Q2
        self.O1 = 0
        self.O2 = 0
        self.G1 = G1
        self.G2 = G2


    def add(self, s):
        if s.rec[1] == self.G1 and self.O1 < self.Q1:
            self.O1 += 1
            self.tuples.append(s)
            return True
        elif s.rec[1] == self.G2 and self.O2 < self.Q2:
            self.O2 += 1
            self.tuples.append(s)
            return True
        return False


    def complete(self):
        if self.O1 == self.Q1 and self.O2 == self.Q2:
            return True
        return False



class Sample:
    def __init__(self, rec, dataset, cost):
        self.rec = rec
        self.dataset_id = dataset
        self.cost = cost


class OptimalBinaryAlg:

    def __init__(self, ds, target, G1=0, G2=1, budget=None):
        self.datasets = {i:ds[i] for i in range(len(ds))}
        self.target = target
        self.G1 = 0
        self.G2 = 1
        if budget != None:
            self.budget = budget
        else:
            self.budget = 50000

   
    def select_group(self, D1, D2):
        # select the dataset of the minority group(self, D1, D2)
        if self.target.O2 == self.target.Q2: 
            return D1
        if self.target.O1 == self.target.Q1: 
            return D2
        if self.datasets[D1].P1 <= self.datasets[D2].P2:
            return D1
        return D2


    def select_dataset(self, group):
        scores = dict()
        if group == self.G1: 
            scores = {i:d.P1/d.C for i, d in self.datasets.items() if d.O1<d.N1}
        if group == self.G2: 
            scores = {i:d.P2/d.C for i, d in self.datasets.items() if d.O2<d.N2}
        return max(scores.items(), key=operator.itemgetter(1))[0]


    def run(self):
        n = len(self.datasets)
        l, cost = 0, 0
        while l < self.budget and not self.target.complete():
            Dls = []
            # find the best dataset for each color
            Dl0 = self.select_dataset(0)
            Dl1 = self.select_dataset(1)
            Dl = self.select_group(Dl0, Dl1)
            Ol = self.datasets[Dl].sample()
            if Ol is not None:
                dec = self.target.add(Ol)
                if dec:
                    # update probs
                    self.datasets[Dl].update_with_sample(Ol)
            cost += self.datasets[Dl].C
            l += 1

        if not self.target.complete():
            print('target %d %d' % (self.target.O1, self.target.O2))
            print('cost %d l %d' % (cost, l))
            return -1, -1
        print('cost %d l %d' % (cost, l))
        return cost, l

