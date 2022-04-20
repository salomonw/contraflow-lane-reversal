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


def eval_obj(tNet2, params):
    tNet = deepcopy(tNet2)
    params['capacity'] = False
    params['max_reversals'] = 0
    params['lambda_cap'] = 0
    #print(params)
    #print({(i, j): tNet.G_supergraph[i][j]['capacity'] for i, j in tNet.G_supergraph.edges()})
    tNet, runtime, _ = cars.solve_bush_CARSn_braess(tNet, **params)
    #print({(i,j): tNet.G_supergraph[i][j]['flow'] for i, j in tNet.G_supergraph.edges()})
    #print(params['userCentric'])
    if params['userCentric'] == True:
        print({(i, j): tNet.G_supergraph[i][j]['lanes'] for i, j in tNet.G_supergraph.edges()})
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    return obj


def more_and_more(l, tNet3, params):
    #tNet3 = deepcopy(tNet2)
    params['lambda_cap'] = 1000
    params['max_reversals'] = l
    params['capacity'] = True
    tNet3, runtime, c = cars.solve_bush_CARSn_braess(tNet3, **params)
    for i, j in tNet.G_supergraph.edges():
        tNet3.G_supergraph[i][j]['capacity'] = c[(i, j)] + 1
    tNet3, _ = cflow.integralize_inputs_braess(tNet3)
    #print(c)
    return tNet3




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
    tNet, max_caps = cflow.integralize_inputs_braess(tNet)
    out_dir = 'tmp/'+net_name+'_'+str(g_mult)
    #n = 3

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


    theta_SO, a_SO = get_pwfunction(fcoeffs, n=5, theta_n=40, theta=False, userCentric=False)
    theta_UC, a_UC = get_pwfunction(fcoeffs, n=5, theta_n=40, theta=False, userCentric=True)

    #print(a_SO)
    #print(a_UC)
    #asd
    obj = {}
    for userCent in [False, True, 'mix']:
        obj[userCent] = []

    params = {
        'fcoeffs': tNet.fcoeffs,
        'n': len(a_SO),
        'theta': theta_SO,
        'a': a_SO,
        'exogenous_G': False,
        'rebalancing': False,
        'linear': False,
        'bush': True,
        'capacity': False,
        'integer': True,
        'lambda_cap': 0,
        'od_flows_flag': False,
        'userCentric': False,
        'max_reversals': 0,
        'Theta_Cap': 1
    }

    tNet2 = deepcopy(tNet)
    obj0 = eval_obj(tNet, params)
    print(obj0)

    params['userCentric'] = True
    params['theta'] = theta_UC
    params['a'] = a_UC
    params['n'] = len(a_UC)
    obj1 = eval_obj(tNet, params)
    print(obj1)
   # print(obj1/obj0)
   # '''




    L = [i for i in range(4)]
    print('i \t UC(Z_UC) \t UC(Z_SO) \t SO(Z_SO)')
    for l in L:
        for userCent in [False, True]:
            if userCent:
                params['theta'] = theta_UC
                params['a'] = a_UC
                params['n'] = len(a_UC)
            else:
                params['theta'] = theta_SO
                params['a'] = a_SO
                params['n'] = len(a_SO)
            params['userCentric'] = userCent
            tNet3 = deepcopy(tNet2)
            # Find optimal allocation
            tNet3 = more_and_more(l, tNet3, params)
            # Evaluate objective
            objv = eval_obj(tNet3, params)
            #print(objv)
            # Save in list
            obj[userCent].append(objv)
            # Evaluate UC with SO solution
            if userCent == False:
                #tNet3 = deepcopy(tNet2)
                userCent = True
                params['theta'] = theta_UC
                params['a'] = a_UC
                params['n'] = len(a_UC)
                objv = eval_obj(tNet3, params)
                userCent = 'mix'
                obj[userCent].append(objv)
        print('{:.0f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(l,
                                      obj[True][-1] / obj0,
                                      obj['mix'][-1] / obj0,
                                      obj[False][-1] / obj0
                                            ))
    mkdir_n(out_dir)
    zdump(objs, out_dir + '/objs_num_reversals.pkl')
    zdump(L, out_dir + '/max_reversals_vec.pkl')
