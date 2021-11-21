import pandas as pd
import matplotlib.pyplot as plt
import statistics
from datetime import datetime
import copy
import json
import numpy as np
from os import listdir
from os.path import isfile, join
import unknown
import baseline
import known_approx as kapprox
import mary_optimal as mary


dirs = ['/localdisk1/DOT/flights/2018', '/localdisk1/DOT/flights/2019', '/localdisk1/DOT/flights/2020']
sourcesdir = '/localdisk1/DOT/csv/'
outputjsondir = '/localdisk1/DOT/json/'
outputdir = '/localdisk1/DOT'
alldata = '/localdisk1/DOT/allflights.csv'
# number of runs
nr = 15 
# data set size
n = 5000

def merge_files():
    fs = [join(d, f) for d in dirs for f in listdir(d) if isfile(join(d, f))]
    #dfs = [pd.read_csv(f) for f in fs]
    dfs = []
    for f in fs:
        dfs.append(pd.read_csv(f))
    onedf = pd.concat(dfs)
    onedf.to_csv(outputdir + 'allflights.csv', index=False)
    print(onedf.size)


def split_into_sources():
    onedf = pd.read_csv(alldata)
    # getting airlines
    airlines = onedf['OP_CARRIER_AIRLINE_ID'].unique()
    states = onedf['ORIGIN_STATE_NM'].unique()
    #states.extend(df['ORIGIN_STATE_FIPS'].unique())
        
    states = list(set(states))
    state_map = {states[i]:i for i in range(len(states))}

    values = [i for i in range(len(states))]

    for a in airlines:
        df = onedf[(onedf['OP_CARRIER_AIRLINE_ID'] == a)]
        conditions = [df['ORIGIN_STATE_NM'] == s for s in states]
        df["DEMOGRAPHICS"] = np.select(conditions, values)
        # making the first two columns id and demographics
        df = df.reset_index()
        df.insert(loc=0, column='id', value=df.index)
        df.insert(loc=1, column='Demographics', value=df["DEMOGRAPHICS"])
        f = sourcesdir + str(a) + '.csv'
        df.to_csv(f, index=False)
        print('%d has %d tuples' % (a, df.size))
        u = df['Demographics'].unique()

    return airlines, states 


def read_datasets_unknown(ads): 
    bds = []
    for a in ads:
        d=json.load(open(a))
        bd = unknown.MaryDataset(d['id'], d['data'], d['groups'], 1)
        bds.append(bd)
    return bds



def read_datasets_known(ads): 
    bds = []
    for a in ads:
        d=json.load(open(a, 'r'))
        bd = kapprox.MaryDataset(d['id'], d['data'], d['groups'], 1)
        bds.append(bd)
    return bds



def read_datasets_baseline(ads): 
    bds = []
    for a in ads:
        d=json.load(open(a))
        bd = baseline.MaryDataset(d['id'], d['data'], d['groups'], 1)
        bds.append(bd)
    return bds





def ddt_input(sourcesdir, Gs, outputdir):

    sfs = [f for f in listdir(sourcesdir) if isfile(join(sourcesdir, f))] 
    for i in range(len(sfs)):
        a = sfs[i]
        df = pd.read_csv(join(sourcesdir, a))
        fname = outputdir + a.replace('csv', 'json', -1) 
        json.dump({'id':i, 'groups':Gs, 'data': df.values.tolist()}, open(fname, 'w'))
    print('done creating ddt input')


def run_exploration(sources, Gs, Qs): 
    print('run_exploration')

    explore_is, explore_cs, ts = [],[],[]

    ads = [join(sources, f) for f in listdir(sources) if isfile(join(sources, f))]
    bds_raw = read_datasets_known(ads)

    for j in range(2):
        for i in range(nr):
            print('nr %d' % (j*nr+i))
            bds = copy.deepcopy(bds_raw)
            t = unknown.MaryTarget(Gs, Qs)
            alg = unknown.UnknownAlg(bds, t, Gs, None, budget)
            st = datetime.now()
            cost, iteras, rews, progs = alg.run_exploration_only()
            et = datetime.now()
            elt = (et - st).total_seconds() * 1000
            if cost != -1:
                explore_cs.append(cost)
                explore_is.append(iteras)
                ts.append(elt)

    if len(explore_is) == 0:
        print('no successful run')
    
    results = {'time': ts, 'cost': explore_cs, 'iters': explore_is}
    json.dump(results, open('results/flights_exploration.json', 'w'))

    print('%d out of %d runs are successful.' % (len(explore_cs), nr))
    print('explore - cost: %f iters: %f' % (sum(explore_cs)/float(len(explore_cs)), sum(explore_is)/float(len(explore_is))))

    
    return results


def run_ucb(sources, Gs, Qs): 
    print('run_ucb')

    ucb_cs, ucb_is, ts = [],[],[]

    ads = [join(sources, f) for f in listdir(sources) if isfile(join(sources, f))]
    bds_raw = read_datasets_known(ads)

    for j in range(2):
        for i in range(nr):
            print('nr %d' % (j*nr+i))
            bds = copy.deepcopy(bds_raw)
            t = unknown.MaryTarget(Gs, Qs)
            alg = unknown.UnknownAlg(bds, t, Gs, None, budget)
            st = datetime.now()
            cost, iteras, rews, progs = alg.run_ucb()
            et = datetime.now()
            elt = (et - st).total_seconds() * 1000
            if cost != -1:
                ucb_cs.append(cost)
                ucb_is.append(iteras)
                ts.append(elt)

    
    results = {'time': ts, 'cost': ucb_cs, 'iters': ucb_is}
    json.dump(results, open('results/flights_ucb.json', 'w'))

    print('%d out of %d runs are successful.' % (len(ucb_cs), nr))
    print('ucb - cost: %f iters: %f' % (sum(ucb_cs)/float(len(ucb_cs)), sum(ucb_is)/float(len(ucb_is))))


    return results




def run_exploitation(sources, Gs, Qs): 
    print('run_exploitation')

    exploit_cs, exploit_is, ts = [],[],[]

    ads = [join(sources, f) for f in listdir(sources) if isfile(join(sources, f))]
    bds_raw = read_datasets_known(ads)

    for j in range(2):
        for i in range(nr):
            print('nr %d' % (j*nr+i))
            bds = copy.deepcopy(bds_raw)
            t = unknown.MaryTarget(Gs, Qs)
            alg = unknown.UnknownAlg(bds, t, Gs, None, budget)
            st = datetime.now()
            cost, iteras, rews, progs = alg.run_exploitation_only()
            et = datetime.now()
            elt = (et - st).total_seconds() * 1000
            if cost != -1:
                exploit_cs.append(cost)
                exploit_is.append(iteras)
                ts.append(elt)


    results = {'time': ts, 'cost': exploit_cs, 'iters': exploit_is}
    json.dump(results, open('results/flights_exploitation.json', 'w'))

    print('%d out of %d runs are successful.' % (len(exploit_cs), nr))
    print('exploite - cost: %f iters: %f' % (sum(exploit_cs)/float(len(exploit_cs)), sum(exploit_is)/float(len(exploit_is))))


    return results



def run_known_ddt(sources, Gs, Qs):
    print('run_known_ddt')
    cc_cs, cc_is, ts = [],[], []

    ads = [join(sources, f) for f in listdir(sources) if isfile(join(sources, f))]
    bds_raw = read_datasets_known(ads)

    for j in range(2):
        for i in range(nr):
            print('nr %d' % (j*nr+i))
            t = kapprox.MaryTarget(Gs, Qs)
            bds = copy.deepcopy(bds_raw) 
            alg = kapprox.ApproxAlg(bds, t, Gs, budget)
            st = datetime.now()
            cost, iteras, rews = alg.run_CC()
            et = datetime.now()
            elt = (et - st).total_seconds() * 1000
            print('cost %d iters %d' % (cost, iteras))
            if cost != -1:
                cc_cs.append(cost)
                cc_is.append(iteras)
                ts.append(elt)

    results = {'time': ts, 'cost': cc_cs, 'iters': cc_is}
    json.dump(results, open('results/flights_cc.json', 'w'))

    print('%d out of %d runs are successful.' % (len(cc_cs), nr))
    print('cc - cost: %f iters: %f' % (sum(cc_cs)/float(len(cc_cs)), sum(cc_is)/float(len(cc_is))))

    return results



def run_baseline(sources, Gs, Qs):
    print('run_baseline')
    baseline_cs, baseline_is, ts = [], [],[]

    ads = [join(sources, f) for f in listdir(sources) if isfile(join(sources, f))]
    bds_raw = read_datasets_known(ads)
    
    # baseline 
    for j in range(2):
        for i in range(nr):
            print('nr %d' % (j*nr+i))
            t = baseline.MaryTarget(Gs, Qs)
            bds = copy.deepcopy(bds_raw)
            alg = baseline.BaselineAlg(bds, t, Gs, budget)
            st = datetime.now()
            cost, iteras = alg.run_Baseline()
            et = datetime.now()
            elt = (et - st).total_seconds() * 1000
            if cost != -1:
                baseline_cs.append(cost)
                baseline_is.append(iteras)
                ts.append(elt)
 

    print('%d out of %d runs are successful.' % (len(baseline_cs), nr))
    print('baseline - cost: %f iters: %f' % (sum(baseline_cs)/float(len(baseline_cs)), sum(baseline_is)/float(len(baseline_is))))

    results = {'time': ts, 'cost': baseline_cs, 'iters': baseline_is}
    json.dump(results, open('results/flights_baseline.json', 'w'))

    return results


def plot():
    cc_results = json.load(open('results/flights_cc.json', 'r'))
    baseline_results = json.load(open('results/flights_baseline.json', 'r'))
    #exploit_results = json.load(open('results/flights_exploitation.json', 'r'))
    explore_results = json.load(open('results/flights_exploration.json', 'r'))
    ucb_results = json.load(open('results/flights_ucb.json', 'r'))

    # times
    #print('cc time: ', statistics.mean(cc_results['time']))
    #print('baseline time: ', statistics.mean(baseline_results['time']))
    #print('exploit time: ', statistics.mean(exploit_results['time']))
    #print('explore time: ', statistics.mean(explore_results['time']))
    #print('ucb time: ', statistics.mean(ucb_results['time']))

    
    algs = ['CouponColl', 'Baseline', 'UCB', 'Explore']#, 'Exploit-Only']
    costs, iters = [], []
    # prep stats
    cs = [c for c in cc_results['cost'] if c > -1]
    its = [c for c in cc_results['iters'] if c > -1]
    if len(cs) < 3:
        costs.append(np.nan)
        iters.append(np.nan)
    else:
        costs.append(statistics.mean(cs))
        iters.append(statistics.mean(its))
    cs = [c for c in baseline_results['cost'] if c > -1]
    its = [c for c in baseline_results['iters'] if c > -1]
    if len(cs) < 20:
        print('insufficient data points')
    if len(cs) < 3:
        costs.append(np.nan)
        iters.append(np.nan)
    else:
        costs.append(statistics.mean(cs))
        iters.append(statistics.mean(its))
    cs = [c for c in ucb_results['cost'] if c > -1]
    its = [c for c in ucb_results['iters'] if c > -1]
    if len(cs) < 3:
        costs.append(np.nan)
        iters.append(np.nan)
    else:
        costs.append(statistics.mean(cs))
        iters.append(statistics.mean(its))
    cs = [c for c in explore_results['cost'] if c > -1]
    its = [c for c in explore_results['iters'] if c > -1]
    if len(cs) < 3:
        costs.append(np.nan)
        iters.append(np.nan)
    else:
        costs.append(statistics.mean(cs))
        iters.append(statistics.mean(its))


    #plot for n's
    font = {'size'   : 15}
    plt.rc('font', **font)
    fig, ax1 = plt.subplots()

    width = 0.25
    xs = [i for i in range(len(algs))]
    ax1.set_xticks(xs) 
    ax1.set_xticklabels(algs)
    #ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Cost')
    ax1.set_yscale('log')

    palette = plt.get_cmap('Set2')
    plt1 = ax1.bar(xs, np.array(costs), width, color=palette(2))
   
    ax2 = ax1.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('#Samples')
    
    plt2 = ax2.plot(xs, np.array(iters), width, color=palette(1), linestyle='--')
   
    fig.tight_layout()
    plt.savefig('plots/flights.pdf')
    print('plots/flights.pdf')
    plt.clf()
    plt.close()


def stats():
    for f in listdir(sourcesdir): 
        if isfile(join(sourcesdir, f)):
            df = pd.read_csv(join(sourcesdir, f))
            print('%s: %d' % (f, df.size))



#merge_files()
#airlines, states = split_into_sources()
#print('split into %d files' % len(airlines))
#json.dump(states, open('/localdisk1/DOT/states', 'w'))
states = json.load(open('/localdisk1/DOT/states', 'r'))
Qs = [int(n/len(states)) for s in states]
Gs = [i for i in range(len(states))]
budget = 400 * sum(Qs)
#ddt_input(sourcesdir, Gs, outputjsondir)
#run_known_ddt(outputjsondir, Gs, Qs)
#run_baseline(outputjsondir, Gs, Qs)
#run_exploitation(outputjsondir, Gs, Qs)
#run_exploration(outputjsondir, Gs, Qs)
#run_ucb(outputjsondir, Gs, Qs)
plot()
#stats()

