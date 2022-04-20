import numpy as np

import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc
plt.style.use(['science','ieee','high-vis'])

def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs


if __name__ == "__main__":
    # PARAMETERS
    net_name = str(sys.argv[1])
    g_mult = 0
    # Read network
    tNet, fcoeffs = read_net(net_name)
    tNet.build_supergraph(identical_G=True)
    # Multiply demand
    g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
    tNet.set_g(g_per)
    # Preprocessing
    tNet, max_caps = cflow.integralize_inputs(tNet)
    out_dir = 'results/'+net_name+'_'+str(g_mult)
    tmp_dir = 'tmp/'+net_name+'_'+str(g_mult)
    n_lines_CARS = 11

    theta, a, rms = cars.get_approx_fun(fcoeffs=tNet.fcoeffs, nlines=n_lines_CARS, range_=[0, 2.5], plot=False)

    objs = {}
    tNet0 = deepcopy(tNet)
    mkdir_n(out_dir)
    mkdir_n(tmp_dir)
    print('--------------------------------------------------------')
    print('Method \t Obj (%) \t Run time (ms) \t Max Flow \t Total Cap')
    print('--------------------------------------------------------')
    algs = ['Nom','MIQP', 'MILP']#, 'LP', 'QP']
    mults = np.linspace(0,1,30)
    results = {}
    for alg in algs:
        results[alg] = []

    for g_mult in mults:
        #g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
        g_per[(1,9)] = 15000 * g_mult
        g_per[(9,1)] = 15000 * (1-g_mult)
        #print(g_per)
        tNet.set_g(g_per)
        tNet0.set_g(g_per)
        objs[g_mult] = []


        params = {
            'tnet': tNet0,
            'fcoeffs': tNet0.fcoeffs,
            'n': n_lines_CARS,
            'theta': theta,
            'a': a,
            'exogenous_G': False,
            'rebalancing': False,
            'linear': False,
            'bush': True,
            'capacity': True,
            'lambda_cap': 1e4,
            'integer': True,
            'max_reversals': 9999
        }

        for alg in algs:
            tNet_ = deepcopy(tNet0)
            params_ = deepcopy(params)
            params_['tnet'] = tNet_
            if alg == 'Nom':
                params_['capacity'] = False
            elif alg == 'MIQP':
                params_ = params_
            elif alg == 'QP':
                params_['integer'] = False
            elif alg == 'MILP':
                params_['linear'] = True
            elif alg == 'LP':
                params_['linear'] = True
                params_['integer'] = False

            if alg in ['Nom', 'MIQP', 'QP', 'MILP', 'LP']:
                tNet_, runtime, od_flows, c = cars.solve_bush_CARSn(**params_)

            if alg =='FW':
                tNet_, obj, TT, dnorm, RG, c, runtime = cflow.solve_FW(tNet_, step=alg, n_iter=4000)
                for (i,j), cap in c.items():
                    tNet_.G_supergraph[i][j]['capacity'] = cap
                    tNet_.G_supergraph[i][j]['lanes'] = cap/1500

            elif alg =='Alt1':
                tNet_, obj1b1, runtime = cflow.solve_alternating(tNet_, e=1e-2, type_='one by one',
                                                             n_lines_CARS=n_lines_CARS, theta=theta, a=a)
            elif alg =='Alt5':
                tNet_, obj5, runtime = cflow.solve_alternating(tNet_, e=1e-2, type_=5,
                                                             n_lines_CARS=n_lines_CARS, theta=theta, a=a)
            elif alg =='AltF':
                tNet_, objfull, runtime = cflow.solve_alternating(tNet_, e=1e-2, type_='full',
                                                              n_lines_CARS=n_lines_CARS, theta=theta, a=a)
            params_ = deepcopy(params)
            params_['capacity'] = False
            params_['linear'] = False
            tNet_ = cflow.project_fluid_sol_to_integer(tNet_)
            params_['tnet'] = tNet_
            tNet_, _, od_flows, c = cars.solve_bush_CARSn(**params_)
            tot_cap = sum(tNet_.G_supergraph[i][j]['capacity'] for i,j in tNet_.G_supergraph.edges())
            max_flow = max([tNet_.G_supergraph[i][j]['flow'] / tNet_.G_supergraph[i][j]['capacity'] for i, j in tNet_.G_supergraph.edges()])

            obj = tnet.get_totalTravelTime(tNet_.G_supergraph, tNet_.fcoeffs)
            if alg == 'Nom':
                nom_obj = obj
            if alg != 'Nom':
                print('{} \t {:.1f} \t {:.2f} \t {:.1f}'.format(alg, obj / nom_obj * 100, runtime, max_flow))

            results[alg].append(100-(obj / nom_obj * 100))
    print(results)
    fig, ax = plt.subplots(figsize=(4, 2))
    for alg in algs:
        if alg != 'Nom':
            ax.plot(mults, results[alg], label=alg, marker='.')

    plt.xlim(min(mults), max(mults))
    plt.xlabel('$\\rho$')
    plt.ylabel('Relative Improvement (\%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('asymmetry_'+net_name+'.pdf')
