import numpy as np
import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc

def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs


def eval_obj(tNet, params):
    params['capacity'] = False
    params['lambda_cap'] = None
    tNet, runtime, od_flows, _ = cars.solve_bush_CARSn(tNet, **params)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    return obj

if __name__ == "__main__":
    # PARAMETERS
    net_name = str(sys.argv[1])
    g_mult = float(sys.argv[2])
    # Read network
    tNet, fcoeffs = read_net(net_name)
    tNet.build_supergraph(identical_G=True)
    # Multiply demand
    g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
    tNet.set_g(g_per)
    # Preprocessing
    tNet, max_caps = cflow.integralize_inputs(tNet)
    out_dir = 'tmp/'+net_name+'_'+str(g_mult)
    tNet2 = deepcopy(tNet)
    tNet3 = deepcopy(tNet)
    n = 7

    theta, a, rms = cars.get_approx_fun(fcoeffs=tNet.fcoeffs, nlines=n, range_=[0, 3], plot=False)

    params = {
        'fcoeffs': tNet.fcoeffs,
        'n': n,
        'theta': theta,
        'a': a,
        'exogenous_G': False,
        'rebalancing': False,
        'linear': False,
        'bush': True,
        'capacity': False,
        'lambda_cap': None
    }

    tNet, runtime, od_flows, c = cars.solve_bush_CARSn(tNet, **params)

    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    print(obj)

    L = np.logspace(0.01,10, num=10)
    for l in L:
        tNet3 = deepcopy(tNet2)
        params['lambda_cap'] = l
        params['capacity'] = True
        tNet3, runtime, od_flows, c = cars.solve_bush_CARSn(tNet3, **params)
        for i, j in tNet.G_supergraph.edges():
            tNet3.G_supergraph[i][j]['capacity'] = c[(i, j)]

        tNet3, _ = cflow.integralize_inputs(tNet3)
        obj = eval_obj(tNet3, params)
        print('Lambda: {}; \t Obj: {}'.format(l, obj))

    #tNet3, obj, TT, dnorm, RG, c = cflow.solve_FW(tNet3, 'FW', n_iter=1000)
    #obj = eval_obj(tNet3, c)
    #print(obj)