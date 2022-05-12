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
    models = ['LP', 'QP', 'MILP', 'MIQP']
    i = 0
    for model in [LP, QP, MILP, MIQP]:
        nLanes = sum([v for k, v in c.items()])/1500
        results.append([net_name, models[i], model.Runtime*1000, model.ObjVal, nLanes, params['n'], model.NumIntVars, model.NumVars - model.NumIntVars, model.NumConstrs])
        i+=1
    return results

if __name__ == "__main__":
    nets = ['test_9', 'EMA', 'SiouxFalls', 'EMA_mid', 'Anaheim', 'NYC']
    cols = ['Network', 'Model', 'Time', 'Objective', 'Num. Lanes', 'L', 'Integer Vars.', 'Continuous Vars.', 'Num. Constraints']
    df = pd.DataFrame()

    for model in ['LP', 'QP', 'MILP', 'MIQP']:

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

                if model == 'MIQP':
                    params['integer'] = True
                    params['linear'] = False
                    _, m, c = cars.solve_bush_CARSn(tNet, **params)
                elif model == 'QP':
                    params['integer'] = False
                    params['linear'] = False
                    _, m, c = cars.solve_bush_CARSn(tNet, **params)
                elif model == 'MILP':
                    params['integer'] = True
                    params['linear'] = True
                    _, m, c = cars.solve_bush_CARSn(tNet, **params)
                elif model == 'LP':
                    params['linear'] = True
                    params['integer'] = False
                    _, m, c = cars.solve_bush_CARSn(tNet, **params)

                nLanes = sum([v for k, v in c.items()]) / 1500

                result = [[net_name, model, m.Runtime * 1000, m.ObjVal, nLanes, params['n'], m.NumIntVars,
                     m.NumVars - m.NumIntVars, m.NumConstrs]]

                #result = solve_and_get_stats(tNet, theta, a)
                df = df.append(pd.DataFrame(result, columns=cols), ignore_index=True)
                #Save results
                print(df)
                df.to_csv('different_net_sizes_exp.csv')

    # Generate plot
    groups = df.groupby('Model')
    # Plot
    fig, ax = plt.subplots(figsize=(4, 3))
    #ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.scatter(group['Num. Lanes'], group['Time'], marker='o', label=name)
    ax.set_xlabel('Num. Lanes in Network')
    ax.set_ylabel('Computational Time (ms)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    ax.legend(frameon=1, facecolor='white', framealpha=0.75, loc=2)
    plt.tight_layout()
    plt.savefig('problem_size.pdf')

