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
        tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
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
    tNet, runtime, _ = cars.solve_bush_CARSn(tNet, **params)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    return obj


def more_and_more(l, tNet3, params):
    #tNet3 = deepcopy(tNet2)
    params['lambda_cap'] = 2
    params['max_reversals'] = l
    params['capacity'] = True
    tNet3, runtime, c = cars.solve_bush_CARSn(tNet3, **params)
    for i, j in tNet.G_supergraph.edges():
        tNet3.G_supergraph[i][j]['capacity'] = c[(i, j)]
    tNet3, _ = cflow.integralize_inputs(tNet3)
    return tNet3

'''
def more_and_more_lanes(tNetc, l, params):
    params['lambda_cap'] = 2
    params['max_reversals'] = l
    params['capacity'] = True

    tNet = deepcopy(tNetc)
    objs = []
    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, params**)
    for m in max_lanes_vec:
        betas = {}
        breaks = {}
        for i, j in tNet.G_supergraph.edges():
            beta0, beta1, breaks0 = cflow.get_arc_pwfunc(tNet, i, j)
            betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}
            breaks[(i, j)] = breaks0
        sol = cflow.solve_opt_int_pwl(tNet, betas=betas, breaks=breaks, max_lanes=m)
        for i, j in tNet.G.edges():
            tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500
            tNetc.G_supergraph[i][j]['t_k'] = cars.travel_time(tNet, i, j)
        obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
        print('max lane changes: ' + str(m)+', obj: ' + str(obj))
        objs.append(obj)
    return objs
'''

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
    # Get piecewise functions
    theta_SO, a_SO = get_pwfunction(fcoeffs, n=9, theta_n=3, theta=False, userCentric=False)
    # Set parameters
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
        'max_reversals': 0
    }
    tNet2 = deepcopy(tNet)
    objs = []
    obj0 = eval_obj(tNet, params)
    L = [i for i in range(30)]
    print('i  \t SO(Z_SO)')
    for l in L:
        tNet3 = deepcopy(tNet2)
        # Find optimal allocation
        tNet3 = more_and_more(l, tNet3, params)
        # Evaluate objective
        objv = eval_obj(tNet3, params)
        # Save in list
        objs.append(objv)
        print('{:.0f} \t {:.3f}'.format(l,objv / obj0))
    mkdir_n(out_dir)
    zdump(objs, out_dir + '/objs_num_reversals.pkl')
    zdump(L, out_dir + '/max_reversals_vec.pkl')





'''
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
    n_lines_CARS = 9

    # SECOND EXPERIMENT (RESTRICT NUMBER OF LANES)
    tNet0 = deepcopy(tNet)
    max_lanes_vec = [i for i in range(30)]
    objs = more_and_more_lanes(tNet0, max_lanes_vec, gmult=g_mult, n_lines_CARS=n_lines_CARS)
    mkdir_n(out_dir)
    zdump(objs, out_dir + '/objs_num_lanes.pkl')
    zdump(max_lanes_vec, out_dir + '/max_lanes_vec.pkl')
'''