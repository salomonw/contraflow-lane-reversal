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


def run_restricted_model(tNet0, betas, breaks, max_lanes=None, max_links=None):
    tNet = deepcopy(tNet0)
    sol = cflow.solve_opt_int_pwl(tNet,
                                  betas=betas,
                                  breaks=breaks,
                                  max_links=max_links,
                                  max_lanes=max_lanes)
    for i, j in tNet.G.edges():
        tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500
        tNet.G_supergraph[i][j]['t_k'] = cars.travel_time(tNet, i, j)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    return tNet, obj

def more_and_more_links(tNetc, max_lanes_vec, gmult=1, n_lines_CARS=5):
    tNet = deepcopy(tNetc)
    objs = []
    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False, rebalancing=False,linear=False, bush=True)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    betas = {}
    breaks = {}
    for i, j in tNet.G_supergraph.edges():
        beta0, beta1, breaks0 = cflow.get_arc_pwfunc(tNet, i, j, plot=0)
        betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}
        breaks[(i, j)] = breaks0
    for m in max_lanes_vec:
        _, objLanes = run_restricted_model(tNet, betas, breaks, max_lanes=m, max_links=None)
        _, objLinks = run_restricted_model(tNet, betas, breaks, max_lanes=None, max_links=m)
        _, obj = run_restricted_model(tNet, betas, breaks, max_lanes=None, max_links=None)
        print('max changes: {} \t linksObj: {} \t lanesObj: {}'.format(m, objLinks, objLanes))
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
    n_lines_CARS = 7
    # Run experiment
    max_lanes_vec = [i for i in range(30)]
    objs = more_and_more_links(tNet, max_lanes_vec, gmult=g_mult, n_lines_CARS=n_lines_CARS)
    mkdir_n(out_dir)
    zdump(objs, out_dir + '/objs_num_links.pkl')
    zdump(max_lanes_vec, out_dir + '/max_links_vec.pkl')
