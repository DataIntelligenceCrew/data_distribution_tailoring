import operator
import sys
import json
import random
from datetime import datetime
random.seed(datetime.now())

class MaryDataset:
    def __init__(self, i, tuples, Gs, cost):
        self.id = i
        self.tuples = tuples
        self.N = len(tuples)
        self.Ns = {j: len([i for i, v in enumerate(tuples) if v[1] == j]) for j in Gs}
        self.Ps = {j: float(self.Ns[j])/self.N for j in Gs}
        self.Os = {j: 0 for j in Gs}
        self.Gs = list(Gs)
        self.C = cost
        self.seen = dict()
        self.Ts = {j: 0 for j in Gs}
        self.t = 0

    
    def sample(self):
        s = Sample(self.tuples[random.randint(0,self.N-1)], self.id, self.C)
        # seen sample
        if s.rec[0] in self.seen:
            return None
        self.seen[s.rec[0]] = True
        return s


    def update_with_sample(self, s):
        j = s.rec[1]
        self.Os[j] += 1
        self.Ps[j] = float(self.Ns[j] - self.Os[j])/self.N

    def update_stats(self, s):
        j = s.rec[1]
        self.Ts[j] += 1



class MaryTarget:

    def __init__(self, Gs, Qs):
        self.Qs = Qs
        self.Gs = Gs
        self.Os = {j: 0 for j in Gs}
        self.tuples = []

    def add(self, s):
        # adding the sampled element to the target
        j = s.rec[1]
        if self.Os[j] < self.Qs[j]:
            self.tuples.append(s)
            self.Os[j] += 1
            return True
        return False


    def complete(self):
        for j in self.Gs:
            if self.Os[j] != self.Qs[j]:
                return False
        return True


class Sample:
    def __init__(self, rec, dataset, cost):
        self.rec = rec
        self.dataset_id = dataset
        self.cost = cost


class OptimalMaryAlg:

    def __init__(self, ds, target, Gs, ps):
        self.datasets = {i:ds[i] for i in range(len(ds))}
        self.target = target
        self.Gs = list(Gs)
        if ps is None:
            self.get_underlying_dist()
        else:
            self.Ps = ps
        self.t = 0
        self.cost = 0
        random.seed(datetime.now())


    def get_underlying_dist(self):
        counts = {j: 0 for j in self.Gs}
        for i in range(len(self.datasets)):
            for s in self.datasets[i].tuples:
                counts[s[1]] += 1
        csum = sum([len(d.tuples) for d in self.datasets.values()])
        self.Ps = {j: float(c)/csum for j, c in counts.items()}


    def select_dataset(self):
        deltas = dict()
        for j in self.Gs:
            # deltaj for each j s.t. Oj<Qj
            deltas[j] = self.exp_cost_next(j)
        # Ci - sum_j^Oj<Qj (delta_j*Pij)
        scores = {i: (self.datasets[i].C - sum([self.datasets[i].Ps[j]*deltas[j] for j in self.Gs if self.target.Os[j] < self.target.Qs[j]])) for i in range(len(self.datasets)) if self.useful(i)}
        # optimal dataset
        if len(scores) == 0:
            print('impossible query')
        return min(scores.items(), key=operator.itemgetter(1))[0]

    
    def useful(self, i):
        for j in self.Gs:
            if self.target.Os[j] < self.target.Qs[j] and self.datasets[i].Ns[j] > self.datasets[i].Os[j]:
                return True
        return False


    def exp_cost_next(self, j):
        scores = dict()
        # choose the best dataset for one of Gj
        for i, d in self.datasets.items():
            # make sure we have not sampled all Gj elements in Di
            # if d.Os[j] == d.Ns[j], the dataset won't be selected
            if d.Os[j] < d.Ns[j]:
                scores[i] = float(d.Ns[j]-d.Os[j])/(d.C*d.N)
        Dj = max(scores.items(), key=operator.itemgetter(1))[0]
        return self.datasets[Dj].C


    def get_reward(self, i):
        return sum([float(self.Ps[j] * self.datasets[i].Ts[j])/(self.datasets[i].C * self.datasets[i].t) for j in self.Gs if self.target.Os[j] < self.target.Qs[j]])


    def run(self):
        Dls, rewards = [], []
        n = len(self.datasets)
        l = 0
        terminate = False
        dupsamples = 0
        while l < 50000 and not terminate: 
            Dl = self.select_dataset()
            # no dataset found and not fulfilled target
            if Dl == -1: 
                return -1, -1, []
            Dls.append(Dl)
            Ol = self.datasets[Dl].sample()
            # update the total number of samples
            self.t += 1
            self.datasets[Dl].t += 1
            # seen sample: discard, still paying the cost
            if Ol is None:
                dupsamples += 1
            else:
                # update only when the sample has not been seen
                self.datasets[Dl].update_stats(Ol)
                dec = self.target.add(Ol) 
                if dec:
                    # updating Os and Ps
                    self.datasets[Dl].update_with_sample(Ol)
                    rewards.append(self.get_reward(Dl))
            self.cost += self.datasets[Dl].C
            l += 1
            if self.target.complete():
                terminate = True
        if not terminate: 
            print('timeout')
        if terminate: 
            print('cost %d l %d dup samples %d' % (self.cost, l, dupsamples))
            return self.cost, l, rewards
        return -1, -1, []



