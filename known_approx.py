import operator
import sys
import json
import random
from datetime import datetime
import copy
random.seed(19)


class MaryDataset:
    def __init__(self, i, tuples, Gs, cost):
        self.id = i
        self.tuples = copy.deepcopy(tuples)
        self.N = len(tuples)
        self.Ns = {j:0 for j in Gs}
        for i, v in enumerate(tuples):
            self.Ns[v[1]] += 1
        self.Ps = {j: float(self.Ns[j])/self.N for j in Gs}
        self.Os = {j: 0 for j in Gs}
        self.Gs = list(Gs)
        self.C = cost
        self.seen = dict()
        self.Ts = {j: 0 for j in Gs}
        self.t = 0


    def group_selected(self, j):
        Os[j] += 1
    
    def sample(self):
        inx = random.randint(0,self.N-1)
        s = Sample(self.tuples[inx], self.id, self.C)
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
    tuples = []

    def __init__(self, Gs, Qs):
        self.Qs = copy.copy(Qs)
        self.Gs = copy.copy(Gs)
        self.Os = {j: 0 for j in Gs}


    def add(self, s):
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

    
    def progress(self):
        fs = [j for j in self.Gs if self.Os[j] == self.Qs[j]]
        return sum(fs)/len(self.Gs)



class Sample:
    def __init__(self, rec, dataset, cost):
        self.rec = rec
        self.dataset_id = dataset
        self.cost = cost


class ApproxAlg:
    def __init__(self, ds, target, Gs, budget=50000):
        self.datasets = {i:copy.deepcopy(ds[i]) for i in range(len(ds))}
        self.target = target
        self.Gs = list(Gs)
        self.changed_groups = {j:True for j in Gs}
        self.Ps = self.get_underlying_dist()
        self.t = 0
        self.Dls, self.Pls = dict(), dict()
        self.budget = budget
        

   
    def get_underlying_dist(self):
        counts = {j: 0 for j in self.Gs}
        for i in range(len(self.datasets)):
            for s in self.datasets[i].tuples:
                counts[s[1]] += 1
        csum = sum([len(d.tuples) for d in self.datasets.values()])
        return {j: float(c)/csum for j, c in counts.items()}




    def select_dataset_group(self, j):
        scores = dict()
        for i, d in self.datasets.items():
            if d.Os[j] < d.Ns[j]:
                scores[i] = (d.Ns[j]-d.Os[j])/float(d.N*d.C)
        if len(scores) > 0:
            return max(scores.items(), key=operator.itemgetter(1))[0]
        else:
            return None


    def rank_datasets(self):
        self.Dls, self.Pls = dict(), dict()
        for j in self.Gs:
            if self.target.Os[j] < self.target.Qs[j]: 
                d = self.select_dataset_group(j)
                if d is not None:
                    self.Dls[j] = d
                    self.Pls[j] = float(self.datasets[d].Ns[j] - self.datasets[d].Os[j])/(self.datasets[d].C * self.datasets[d].N)
        return sorted(self.Pls.items(), key=operator.itemgetter(1)), self.Dls

    
    def select_dataset(self):
        deltas = dict()
        for j in self.Gs:
            # deltaj for each j s.t. Oj<Qj
            deltas[j] = self.exp_cost_next(j)
        # sum_j^Oj<Qj (delta_j*Pij)  / C_i
        scores = {i: (sum([self.datasets[i].Ps[j]*deltas[j] for j in self.Gs if self.target.Os[j] < self.target.Qs[j]])/self.datasets[i].C) for i in range(len(self.datasets)) if self.useful(i)}
        # optimal dataset
        if len(scores) == 0:
            print('impossible query')
        return max(scores.items(), key=operator.itemgetter(1))[0]

    
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
            if d.Os[j] < d.Ns[j]:
                scores[i] = float(d.Ns[j]-d.Os[j])/(d.C*d.N)
        Dj = max(scores.items(), key=operator.itemgetter(1))[0]
        return self.datasets[Dj].C




    def get_reward(self, i):
        return sum([float(self.Ps[j] * self.datasets[i].Ts[j])/(self.datasets[i].C * self.datasets[i].t) for j in self.Gs if self.target.Os[j] < self.target.Qs[j]])

    def run_CC(self):
        rewards = []
        n = len(self.datasets)
        l, cost = 0, 0
        terminate = False
        while l < self.budget and not terminate:
            Pls, Dls = self.rank_datasets()
            if len(Dls) == 0: 
                return -1, -1, []
            Dl = Dls[Pls[0][0]]
            Ol = self.datasets[Dl].sample()
            if Ol is not None:
                dec = self.target.add(Ol) 
                if dec:
                    # updating Os and Ps
                    self.datasets[Dl].update_with_sample(Ol)
            cost += self.datasets[Dl].C
            l += 1
            if self.target.complete():
                terminate = True
        print('cost %d  l %d' % (cost, l))
        if not terminate: 
            print('timeout with %f progress' % self.target.progress())
        if terminate: 
            return cost, l, rewards
        return -1, -1, []


