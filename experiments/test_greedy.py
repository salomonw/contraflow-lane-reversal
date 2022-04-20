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

def more_and_more_links(tNetc, max_lanes_vec, gmult=1, n_lines_CARS=5):
    tNet = deepcopy(tNetc)
    objs = []
    tNet, runtime, od_flows, _ = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False, rebalancing=False,linear=False, bush=True)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    for m in max_lanes_vec:
        betas = {}
        breaks = {}
        for i, j in tNet.G_supergraph.edges():
            beta0, beta1, breaks0 = cflow.get_arc_pwfunc(tNet, i, j)
            betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}
            breaks[(i, j)] = breaks0
        sol = cflow.solve_greedy(tNet, betas, breaks, max_lanes=None, max_links=None)
        for i, j in tNet.G.edges():
            tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500
            tNetc.G_supergraph[i][j]['t_k'] = cars.travel_time(tNet, i, j)
        obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
        print('max links changes: ' + str(m)+', obj: ' + str(obj))
        objs.append(obj)
    return objs


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
    n_lines_CARS = 5

    # SECOND EXPERIMENT (RESTRICT NUMBER OF LINKS)
    tNet0 = deepcopy(tNet)
    g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
    tNet0.set_g(g_per)
    max_lanes_vec = [i for i in range(30)]
    objs = more_and_more_links(tNet0, max_lanes_vec, gmult=g_mult, n_lines_CARS=n_lines_CARS)
    mkdir_n(out_dir)
    zdump(objs, out_dir + '/objs_num_links.pkl')
    zdump(max_lanes_vec, out_dir + '/max_links_vec.pkl')


