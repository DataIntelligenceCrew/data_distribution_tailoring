import random
import json

#ms: groups
#ps: proportion of groups 

def gen_same_dist_dataset(ms, s):
    random.seed(a=None, version=2)
    # appending ids
    # all datasets are sampled from the same distribution
    return [(i, random.choice(ms)) for i in range(s)]


def gen_rand_dist_dataset(ms, s):
    random.seed(a=None, version=2)
    # create a random distribution of groups
    dists = [random.randint(1,len(ms)) for m in ms] 
    dsum = sum(dists)
    sdists = [d/dsum for d in dists]
    # appending ids
    return [(i, random.choices(ms, sdists, k=1)[0]) for i in range(s)]



def gen_one_minority_dist_dataset(ms, s, mingroup=0):
    random.seed(a=None, version=21)
    k = len(ms)
    ps = [1.0/k for m in ms]
    # mingroup is the minority
    for i in range(1,k):
        r = random.uniform(ps[0]/(k*(k-1)), ps[0]/(k-1))
        ps[i] += r
        ps[0] -= r
    print('minority ps')
    # change ps s.t. maingroup is minority
    nmg = ps[mingroup]
    ps[mingroup] = ps[0]
    ps[0] = nmg
    print(ps)
    return [(i, random.choices(ms, ps, k=1)[0]) for i in range(s)]
 


def gen_one_majority_dist_dataset(ms, s):
    random.seed(a=None, version=21)
    k = len(ms)
    ps = [1.0/k for m in ms]
    # the first one is the majority
    for i in range(1,k):
        r = random.uniform(ps[0]/(k*(k-1)), ps[0]/(k-1))
        ps[i] -= r
        ps[0] += r
    print('majority ps')
    print(ps)
    return [(i, random.choices(ms, ps, k=1)[0]) for i in range(s)]





def gen_dist_dataset(ms, ps, s):
    random.seed(a=None, version=2)
    vs = []
    for im in range(len(ms)):
        # adding tuples for each demographic
        ss = [(i+len(vs), m[im]) for i in range(int(ps[im]*s))]
        vs += ss
    return vs


def add_rand_cost(ds, min_cost, max_cost):
    cds = []
    for d in ds:
        d['cost'] = random.randint(min_cost, max_cost)
        cds.append(d)
    return cds


def add_uniform_cost(ds, min_cost, max_cost):
    cost = random.randin(min_cost, max_cost) 
    cds = []
    for d in ds:
        d['cost'] = cost
        cds.append(d)
    return cds


def add_equi_cost(ds):
    # assign unit cost to all data sets
    cds = []
    for d in ds:
        d['cost'] = 1
        cds.append(d)
    return cds




def satisfy_cond(ms, cs, st):
    if cs == None:
        return []
    return [(st+i*j, ms[i])for i in range(len(ms)) for j in range(cs[i])]



def gen_datalake(n, ms, min_row, max_row, min_cost, max_cost, cost=1, datadist='same-dist', costdist='random', mingroups = None, s=None, ps=None, mingroupid=0):

    ds = []
    if s == None: 
        if mingroups == None:
            s = random.randint(min_row, max_row)
        else:
            s = random.randint(min_row - sum(mingroups), max_row - sum(mingroups))

    if datadist == 'random':
        print('datadist = random')
        ds = [{'id': i, 'data': gen_rand_dist_dataset(ms, s) + satisfy_cond(ms, mingroups, s), 'groups': ms} for i in range(n)]


    if datadist == 'same':
        print('datadist = same')
        # datasets with same distbn
        ds = [{'id': i, 'data': gen_same_dist_dataset(ms, s) + satisfy_cond(ms, mingroups, s), 'groups': ms} for i in range(n)]

    if datadist == 'minority': 
        print('datadist = minority')
        ds = [{'id': i, 'data': gen_one_minority_dist_dataset(ms, s), 'groups': ms} for i in range(n)]

    if datadist == 'minority-randgroup': 
        print('datadist = minority-randgroup')
        ds = [{'id': i, 'data': gen_one_minority_dist_dataset(ms, s, random.choice(ms)), 'groups': ms} for i in range(n)]

    if datadist == 'majority':
        print('datadist = majority')
        ds = [{'id': i, 'data': gen_one_majority_dist_dataset(ms, s), 'groups': ms} for i in range(n)]

    if datadist == 'minmajorityratio':
        print('datadist = minmajorityratio')
        # ps is the percenntage of minority and majority data sets
        ds = [{'id': i, 'data': gen_one_majority_dist_dataset(ms, s), 'groups': ms} for i in range(int(n*ps[0]))]
        j = len(ds)
        ds.extend([{'id': j+i, 'data': gen_one_minority_dist_dataset(ms, s), 'groups': ms} for i in range(int(n*ps[1]))])


    if costdist == 'random':
        cds = add_rand_cost(ds, min_cost, max_cost)

    if costdist == 'uniform':
        cds = add_uniform_cost(ds, min_cost, max_cost)
    if costdist == 'equi':
        cds = add_equi_cost(ds)
    
    return cds


# n, ms, min_row, max_row, min_cost, max_cost, cost=1, dist='random', cost='random', s=None, ps=None
#ds = gen_datalake(3, 5, [0,1], 2, 5, 1, 3, None, 'random', 'random', ps=None)
#print(ds)
#json.dump(ds, open('data_synthetic/binary_repo.json', 'w'))


