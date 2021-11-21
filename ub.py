import math
import operator
import sys
import json
import random
from datetime import datetime

class MaryDataset:
    C = 1
    def __init__(self, i, tuples, Gs, cost):
        self.id = i
        self.tuples = tuples
        self.N = len(tuples)
        self.Ns = {j: len([i for i, v in enumerate(tuples) if v[1] == j]) for j in Gs}
        self.Os = {j: 0 for j in Gs}
        self.Gs = Gs
        self.C = cost
        self.seen = {}

    def group_selected(self, j):
        Os[j] += 1
    
    def sample(self):
        random.seed(datetime.now())
        s = Sample(self.tuples[random.randint(0,self.N-1)], self.id, self.C)
        j = s.rec[1]
        # seen sample
        if s.rec[0] in self.seen:
            return None
        self.seen[s.rec[0]] = True

        return s


class MaryTarget:
    # iteration id
    l = 0
    cost = 0
    tuples = []

    def __init__(self, Gs, Qs):
        self.Qs = Qs
        self.Gs = Gs
        self.Os = {j: 0 for j in Gs}


    def add(self, s, j):
        j = s.rec[1]
        if s.rec[1] == j:
            self.tuples.append(s)
            self.Os[j] += 1
            return True
        return False


    def complete(self):
        for j in self.Gs:
            if self.Os[j] != self.Qs[j]:
                return False
        print('completed successfully!')
        return True


class Sample:
    def __init__(self, rec, dataset, cost):
        self.rec = rec
        self.dataset_id = dataset
        self.cost = cost


class CouponCollectorAlg:
    cost = 0

    def __init__(self, ds, target, Gs):
        self.datasets = {i:ds[i] for i in range(len(ds))}
        self.target = target
        self.Gs = Gs
    

    def select_dataset_group(self, j):
        # choosing the best dataset for group j
        scores = dict()
        for i, d in self.datasets.items():
            # double checking 
            if self.target.Qs[j] <= d.Ns[j]:
                # cost per unit
                scores[i] = (d.N*d.C)/d.Ns[j]
        return min(scores.items(), key=operator.itemgetter(1))[0]


    def compute_theo_eub(self):
        for j in self.target.Gs:
            # select the best dataset for each group 
            Dl = self.select_dataset_group(j)
            # currently, datasets are guaranteed to have Qj elements
            if self.target.Qs[j] > self.datasets[Dl].Ns[j]:
                print('not a good dataset selected.')
            self.cost += (self.datasets[Dl].C * self.datasets[Dl].N)*math.log(self.datasets[Dl].Ns[j]/(self.datasets[Dl].Ns[j]-self.target.Qs[j]))
        print('eub cost %d' % (self.cost))
        return self.cost 




    def run(self):
        l = 0
        for j in self.target.Gs:
            dupsamples = 0
            # select the best dataset for each group 
            Dls = self.select_dataset_group(j)
            Dl = -1
            for d in Dls:
                if self.target.Qs[j] <= self.datasets[d[0]].Ns[j]:
                    Dl = d[0]
                    break
                else:
                    print('considering next best')
            if Dl == -1:
                # currently, datasets are guaranteed to have Qj elements
                print('not a good dataset selected.')
                continue
            # keep sampling from the best dataset
            # committing to the dataset until we get all Qj
            while self.target.Os[j] < self.target.Qs[j]:
                Ol = self.datasets[Dl].sample()
                # seen sample, still counting its cost
                if Ol is None:
                    dupsamples += 1
                # unseen sample
                else:
                    dec = self.target.add(Ol, j)
                self.cost += self.datasets[Dl].C
                l += 1
                # timeout condition
                if l > max(50000, self.datasets[Dl].N):
                    print('timeout')
                    return -1, -1 
            print('cost %d l %d dupsamples %d' % (self.cost, l, dupsamples))
            if self.target.Os[j] < self.target.Qs[j]:
                print('failed to build the dataset')
                return -1, -1
        print('cost %d l %d' % (self.cost, l))
        return self.cost, l



