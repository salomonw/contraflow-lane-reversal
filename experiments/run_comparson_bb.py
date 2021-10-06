import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc
from datetime import datetime

plt.style.use(['science','ieee', 'high-vis'])

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
    g_mult = float(sys.argv[2])
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
    n_lines_CARS = 5



    objs = {}
    tNet0 = deepcopy(tNet)
    mkdir_n(out_dir)
    mkdir_n(tmp_dir)
    g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
    tNet.set_g(g_per)
    # print(min([tNet.G_supergraph[i][j]['max_capacity'] for i,j in tNet.G_supergraph.edges()]))
    objs[g_mult] = []
    objs_labels = ['Nominal']
    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False,
                                                    rebalancing=False, linear=False, bush=True)
    # print(min([tNet.G_supergraph[i][j]['max_capacity'] for i, j in tNet.G_supergraph.edges()]))
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    objs[g_mult].append(obj)
    print('Nominal:' + str(obj))

    tNet0 = deepcopy(tNet)
    betas = {}
    breaks = {}
    for i, j in tNet.G_supergraph.edges():
        beta0, beta1, breaks0 = cflow.get_arc_pwfunc(tNet, i, j)
        betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}
        breaks[(i, j)] = breaks0

    t = datetime.now()
    sol = cflow.solve_opt_int_pwl(tNet, betas=betas, breaks=breaks, max_lanes=len(tNet.G_supergraph.edges()))
    t = datetime.now() - t
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    print('Linear   \t Obj: {} \t Time= {}'.format(obj,t))

    t = datetime.now()
    sol = cflow.solve_opt_int_pwl(tNet0, betas=betas, breaks=breaks, max_lanes=len(tNet0.G_supergraph.edges()), binary=True),
    t = datetime.now() - t
    obj = tnet.get_totalTravelTime(tNet0.G_supergraph, tNet0.fcoeffs)
    print('Integer \t Obj: {} \t Time= {}'.format(obj,t))

