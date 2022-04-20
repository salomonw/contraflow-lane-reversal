import numpy as np
import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc
import pandas as pd

def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs

# Get piecewise functions
def get_pwfunction(fcoeffs, n, theta_n, theta=False, userCentric=False):
    fc = fcoeffs.copy()
    if userCentric:
        fc = cars.UC_fcoeffs(fc)
    if theta == False:
        theta, a, rms = cars.get_approx_fun(fcoeffs=fc, nlines=n, range_=[0, theta_n], plot=False)
    else:
        theta, a, rms = cars.get_approx_fun(fcoeffs=fc, nlines=n, range_=[0, theta_n], theta=theta, plot=False)
    return theta, a

def solve_and_get_stats(tNet, theta, a):
    # Define parameters
    params = {
        'fcoeffs': tNet.fcoeffs,
        'n': len(a),
        'theta': theta,
        'a': a,
        'exogenous_G': False,
        'rebalancing': False,
        'linear': False,
        'bush': True,
        'capacity': True,
        'integer': True,
        'lambda_cap': 1,
        'od_flows_flag': False,
        'userCentric': False,
        'max_reversals': 50000
    }

    _, MIQP, c = cars.solve_bush_CARSn(tNet, **params)
    params['integer'] = False
    tNet2, QP, c = cars.solve_bush_CARSn(tNet, **params)
    tNet3 = cflow.integralize_inputs(tNet2)

    params['integer'] = True
    params['linear'] = True
    _, MILP, c = cars.solve_bush_CARSn(tNet, **params)
    params['integer'] = False
    tNet2, LP, c = cars.solve_bush_CARSn(tNet, **params)



    results = []
    models = ['MIQP', 'QP', 'MILP', 'LP']
    i = 0
    for model in [MIQP, QP, MILP, LP]:
        nLanes = sum([v for k, v in c.items()])/1500
        results.append([net_name, models[i], model.Runtime*1000, model.ObjVal, nLanes, params['n'], model.NumIntVars, model.NumVars - model.NumIntVars, model.NumConstrs])
        i+=1
    return results

if __name__ == "__main__":
    nets = ['test_9', 'EMA', 'SiouxFalls', 'EMA_mid', 'Anaheim', 'NYC']
    nets = ['test_9', 'EMA', 'SiouxFalls', 'EMA_mid', 'Anaheim']
    cols = ['Network', 'Model', 'Time', 'Objective', 'Num. Lanes', 'L', 'Integer Vars.', 'Continuous Vars.', 'Num. Constraints']
    df = pd.DataFrame()
    for net_name in nets:
        # Read net
        tNet, fcoeffs = read_net(net_name)
        # Create supergraph
        tNet.build_supergraph(identical_G=True)
        # Multiply demand
        g_mult = 2
        g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
        tNet.set_g(g_per)
        # Preprocessing
        tNet, max_caps = cflow.integralize_inputs(tNet)

        for l in [3, 5, 7]:
            # find different granularities
            theta, a = get_pwfunction(fcoeffs, n=l, theta_n=3, theta=False, userCentric=False)
            result = solve_and_get_stats(tNet, theta, a)
            df = df.append(pd.DataFrame(result, columns=cols), ignore_index=True)
    df.to_csv('different_net_sizes_exp.csv')