import operator
import sys
import json
import random
from datetime import datetime
import copy
from scipy.spatial import distance


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

    def sample(self):
        random.seed(datetime.now())
        inx = random.randint(0,self.N-1)
        s = Sample(self.tuples[inx], self.id, self.C)
        if s.rec[0] in self.seen:
            return None
        self.seen[s.rec[0]] = True
        return s


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

class Sample:
    def __init__(self, rec, dataset, cost):
        self.rec = rec
        self.dataset_id = dataset
        self.cost = cost


class BaselineAlg:
    def __init__(self, ds, target, Gs, termination):
        self.datasets = {i:copy.deepcopy(ds[i]) for i in range(len(ds))}
        self.target = target
        self.Gs = list(Gs)
        if termination is None:
            self.term = 500 * sum(self.target.Qs)
        else:
            self.term = termination

    def select_dataset(self):
        random.seed(datetime.now())
        return  random.randint(0,len(self.datasets)-1)

    def distance_from_goal(self):
        q = [self.target.Qs[i] for i in self.target.Gs]
        o = [self.target.Os[i] for i in self.target.Gs]
        return distance.jensenshannon(q, o)



    def run_Baseline(self):
        # sample up to sum Qs
        # if not terminate sample twice what is not satisfied
        n = len(self.datasets)
        iters, cost = 0, 0
        terminate = False
        budget = 2 * max([self.target.Qs[j]-self.target.Os[j] for j in self.target.Gs])
        while not terminate: 
            if budget > self.term: 
                print('timeout!')
                return -1, -1
            budget = 2 * max([self.target.Qs[j]-self.target.Os[j] for j in self.target.Gs])
            l = 0
            while l < budget and not terminate:
                Dl = self.select_dataset()
                Ol = self.datasets[Dl].sample()
                if Ol is not None:
                    dec = self.target.add(Ol) 
                cost += self.datasets[Dl].C
                l += 1
                if self.target.complete():
                    print('terminated with %d iterations of %d cost' % (cost, iters))
                    terminate = True
            iters += budget 
            if terminate: 
                return cost, iters




    def run_RandomSampling(self):
        n = len(self.datasets)
        l, cost = 0, 0
        terminate = False
        while l < self.budget and not terminate:
            Dl = self.select_dataset()
            Ol = self.datasets[Dl].sample()
            if Ol is not None:
                dec = self.target.add(Ol) 
            cost += self.datasets[Dl].C
            l += 1
            if self.target.complete():
                terminate = True
        print('cost %d  l %d' % (cost, l))
        if not terminate: 
            print('timeout')
        if terminate: 
            return cost, l, 0
        dist = self.distance_from_goal()
        return -1, -1, dist


