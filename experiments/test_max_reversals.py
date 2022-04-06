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
    params['lambda_cap'] = 2
    tNet, runtime, od_flows, _ = cars.solve_bush_CARSn(tNet, **params)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    return obj


def more_and_more(l, tNet2, params, theta_UC, a_UC):
    tNet3 = deepcopy(tNet2)
    params['lambda_cap'] = 2
    params['max_reversals'] = l
    params['capacity'] = True
    tNet3, runtime, od_flows, c = cars.solve_bush_CARSn(tNet3, **params)
    for i, j in tNet.G_supergraph.edges():
        tNet3.G_supergraph[i][j]['capacity'] = c[(i, j)]
    tNet3, _ = cflow.integralize_inputs(tNet3)

    # Evaluation
    params['max_reversals'] = 0
    params['lambda_cap'] = 0
    #tNet3, runtime, od_flows, c = cars.solve_bush_CARSn(tNet3, **params)
    obj = eval_obj(tNet3, params)
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
    n = 5

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


    theta_SO, a_SO = get_pwfunction(fcoeffs, n, theta_n=3, theta=False, userCentric=False)
    theta_UC, a_UC = get_pwfunction(fcoeffs, n, theta_n=3, theta=False, userCentric=True)

    obj = {}
    for userCent in [True, False]:
        obj[userCent] = []

    params = {
        'fcoeffs': tNet.fcoeffs,
        'n': n,
        'theta': theta_SO,
        'a': a_SO,
        'exogenous_G': False,
        'rebalancing': False,
        'linear': False,
        'bush': True,
        'capacity': False,
        'integer': True,
        'lambda_cap': 2,
        'od_flows_flag': True,
        'userCentric': True,
        'max_reversals': 0
    }

    tNet, runtime, od_flows, c = cars.solve_bush_CARSn(tNet, **params)

    obj0 = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    print(obj0)

    L = [i for i in range(30)]
    print('i \t UC \t SO')
    for l in L:
        for userCent in [True, False]:
            if userCent:
                params['theta'] = theta_UC
                params['a'] = a_UC
            else:
                params['theta'] = theta_SO
                params['a'] = a_SO
            params['userCentric'] = userCent
            objv = more_and_more(l, tNet2, params, theta_UC, a_UC)
            obj[userCent].append(objv)
        print('{} \t {} \t {}'.format(l, obj[True][-1] / obj0, obj[False][-1] / obj0))
    mkdir_n(out_dir)
    zdump(objs, out_dir + '/objs_num_reversals.pkl')
    zdump(L, out_dir + '/max_reversals_vec.pkl')
