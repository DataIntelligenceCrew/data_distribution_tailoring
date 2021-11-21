import operator
import statistics
import sys
import json
import random
import math
from datetime import datetime

class MaryDataset:
    C = 1
    def __init__(self, i, tuples, Gs, cost):
        self.id = i
        self.tuples = tuples
        self.N = len(tuples)
        # number of samples taken from each group
        self.Ts = {j: 0 for j in Gs}
        self.Gs = Gs
        self.C = cost
        # number of samples taken so far
        self.t = 0 
        self.seen = dict()
        # true probs for computing regret
        self.Ns = {j:0 for j in Gs}
        for i, v in enumerate(tuples):
            self.Ns[v[1]] += 1
        self.Ps = {j: float(self.Ns[j])/self.N for j in Gs}
        

    def sample(self):
        random.seed(datetime.now())
        s = Sample(self.tuples[random.randint(0,self.N-1)], self.id, self.C)
        # seen sample
        if s.rec[0] in self.seen:
            return None
        self.seen[s.rec[0]] = True
        return s

    
    def update_stats(self, s):
        j = s.rec[1]
        self.Ts[j] += 1 



class MaryTarget:
    tuples = []

    def __init__(self, Gs, Qs):
        self.Qs = Qs
        self.Gs = Gs
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


class UnknownAlg:

    def __init__(self, ds, target, Gs, ps=None, budget=50000):
        self.datasets = {i:ds[i] for i in range(len(ds))}
        self.target = target
        self.Gs = Gs
        # number of samples taken so far
        self.t = 0  
        # underlying distribution
        if ps is None:
            self.get_underlying_dist()
        else:
            self.Ps = ps
        self.budget = budget


    def get_underlying_dist(self):
        counts = {j: 0 for j in self.Gs}
        for i in range(len(self.datasets)):
            for s in self.datasets[i].tuples:
                counts[s[1]] += 1
        csum = sum([len(d.tuples) for d in self.datasets.values()])
        self.Ps = {j: float(c)/csum for j, c in counts.items()}


            
    def select_exploit_dataset(self):
        rewards = {i: self.get_reward(i) for i in range(len(self.datasets))}
        Dl = max(rewards.items(), key=operator.itemgetter(1))[0]
        return Dl


    def select_dataset(self):
        rewards = dict()
        for i in range(len(self.datasets)):
            #reward of a dataset
            rewards[i] = self.get_reward(i)
            # upper bound based on UCB strategy and Hoeffding’s Inequality
            ub = self.get_upper_bound(i)
            rewards[i] += ub
        Dl = max(rewards.items(), key=operator.itemgetter(1))[0]
        return Dl


    def get_upper_bound(self, i):
        a = 0.0
        b = max([self.Ps[j]/self.datasets[i].C for j in self.Gs if self.target.Qs[j] > self.target.Os[j]])
        return (b-a) * math.sqrt(2.0*math.log(self.t)/self.datasets[i].t)


    def first_round_sample(self):
        rewards = []
        cost = 0
        for Dl in range(len(self.datasets)):
            Ol = self.datasets[Dl].sample()
            # update the total number of samples
            self.t += 1
            self.datasets[Dl].t += 1
            if Ol is not None:
                # update only when the sample has not been seen
                self.datasets[Dl].update_stats(Ol)
                dec = self.target.add(Ol)
                if dec:
                    rewards.append(self.get_reward(Dl))
                else: 
                    rewards.append(0.0)
            cost += self.datasets[Dl].C
        return rewards, cost


    
    def get_reward(self, i):
        return sum([float(self.datasets[i].Ts[j])/(self.Ps[j]*self.datasets[i].C * self.datasets[i].t) for j in self.Gs if self.target.Os[j] < self.target.Qs[j]])


    def run_exploitation_only(self):
        progress = []
        terminate = False
        dupsamples = 0
        rewards, cost = self.first_round_sample()
        # consider one round of sampling datasets
        l = len(self.datasets)
        if self.target.complete():
            terminate = True
        # pick the best dataset  for exploitation
        Dl = self.select_exploit_dataset()
        while l < self.budget and not terminate:
            # getting rewards
            Ol = self.datasets[Dl].sample()
            # update the total number of samples
            self.t += 1
            self.datasets[Dl].t += 1
            if Ol is None:
                dupsamples += 1
            else:
                # update only when the sample has not been seen
                self.datasets[Dl].update_stats(Ol)
                dec = self.target.add(Ol)
                if dec: 
                    rewards.append(self.get_reward(Dl))
                else: 
                    rewards.append(0.0)
            cost += self.datasets[Dl].C
            l += 1
            if self.target.complete():
                terminate = True
        if not terminate:
            print('timeout')
        if terminate:
            print('cost %d l %d dupsamples %d' % (cost, l, dupsamples))
            return cost, l, rewards, progress
        return  -1, -1, [], []




    def run_ucb(self):
        progress = []
        n = len(self.datasets)
        terminate = False
        dupsamples = 0
        rewards, cost = self.first_round_sample()
        # consider one round of sampling datasets
        l = len(self.datasets)
        if self.target.complete():
            terminate = True
        while l < self.budget and not terminate:
            Dl = self.select_dataset()
            Ol = self.datasets[Dl].sample()
            # update the total number of samples
            self.t += 1
            self.datasets[Dl].t += 1
            if Ol is None:
                dupsamples += 1
            else:
                # update only when the sample has not been seen
                self.datasets[Dl].update_stats(Ol)
                dec = self.target.add(Ol)
                if dec:
                    rewards.append(self.get_reward(Dl))
                else:
                    rewards.append(0.0)
            cost += self.datasets[Dl].C
            l += 1
            if self.target.complete():
                terminate = True
        if not terminate:
            print('timeout')
        if terminate:
            print('cost %d l %d dupsamples %d' % (cost, l, dupsamples))
            return cost, l, rewards, progress
        return -1, -1, [], []

    

    def select_dataset_cost(self):
        scores = {i: 1.0/self.datasets[i].C for i in range(len(self.datasets))}
        ssum = sum(list(scores.values()))
        ps, ls = [], []
        for l, s in scores.items():
            ls.append(l)
            ps.append(s/ssum)
        random.seed(datetime.now())
        return random.choices(ls, ps, k=1)[0]



    def run_exploration_only(self):
        print('run_exploration_only')
        progress, rewards = [], []
        n = len(self.datasets)
        l, cost = 0, 0
        terminate = False
        dupsamples = 0
        while l < self.budget and not terminate:
            Dl = self.select_dataset_cost()
            Ol = self.datasets[Dl].sample()
            # update the total number of samples
            self.t += 1
            self.datasets[Dl].t += 1
            if Ol is None:
                dupsamples += 1
                rewards.append(0.0)
            else:
                dec = self.target.add(Ol)
                self.datasets[Dl].update_stats(Ol)
                if dec: 
                    rewards.append(self.get_reward(Dl))
                else:
                    rewards.append(0.0)
            cost += self.datasets[Dl].C
            l += 1
            if self.target.complete():
                terminate = True
        if not terminate:
            print('timeout')
        if terminate:
            print('cost %d l %d dupsamples %d' % (cost, l, dupsamples))
            return cost, l, rewards, progress
        return -1, -1, [], []



