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
    tNet2 = deepcopy(tNet)

    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs,
                                                    n=n_lines_CARS, exogenous_G=False,
                                                    rebalancing=False, linear=False, bush=True,
                                                    capacity=False)

    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    print(obj)

    tNet2, runtime, od_flows = cars.solve_bush_CARSn(tNet2, fcoeffs=tNet2.fcoeffs,
                                                    n=n_lines_CARS, exogenous_G=False,
                                                    rebalancing=False, linear=False, bush=True,
                                                    capacity=True)

    obj = tnet.get_totalTravelTime(tNet2.G_supergraph, tNet2.fcoeffs)
    print(obj)