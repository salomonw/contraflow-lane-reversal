import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc
import numpy as np

def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs

def solve_SO(tNet0, g, theta, a, MSA=False):
    tNet = deepcopy(tNet0)
    g_per = tnet.perturbDemandConstant(tNet.g, g)
    tNet.set_g(g_per)
    if MSA:
        tNet.solveMSA_social()
        obj = tnet.get_totalTravelTime(tNet.G, tNet.fcoeffs)
    else:
        tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet,
                                                        fcoeffs=tNet.fcoeffs,
                                                        n=10,
                                                        exogenous_G=False,
                                                        rebalancing=False,
                                                        linear=False,
                                                        bush=True,
                                                        theta=theta,
                                                        a=a,
                                                        userCentric=False)
        cong = [tNet.G_supergraph[i][j]['flow']/tNet.G_supergraph[i][j]['capacity'] for i,j in tNet.G_supergraph.edges()]
        cong_avg = np.mean(cong)
        cong_max = max(cong)
        #print('SO: Cong Avg:{};\t Cong Max:{}'.format(cong_avg, cong_max))
        obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    return obj

def solve_UE(tNet0, g, theta, a, MSA=False):
    tNet = deepcopy(tNet0)
    g_per = tnet.perturbDemandConstant(tNet.g, g)
    tNet.set_g(g_per)
    if MSA:
        tNet.solveMSA()
        obj = tnet.get_totalTravelTime(tNet.G, tNet.fcoeffs)
    else:
        tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet,
                                                        fcoeffs=tNet.fcoeffs,
                                                        n=10,
                                                        exogenous_G=False,
                                                        rebalancing=False,
                                                        linear=False,
                                                        bush=True,
                                                        theta=theta,
                                                        a=a,
                                                        userCentric=True)

        cong = [tNet.G_supergraph[i][j]['flow']/tNet.G_supergraph[i][j]['capacity'] for i,j in tNet.G_supergraph.edges()]
        cong_avg = np.mean(cong)
        cong_max = max(cong)
        print('UE: Cong Avg:{};\t Cong Max:{}'.format(cong_avg, cong_max))
        obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)

    return obj


if __name__ == "__main__":
    # PARAMETERS
    net_name = str(sys.argv[1])

    # Read network
    tNet, fcoeffs = read_net(net_name)
    tNet.build_supergraph(identical_G=True)
    #out_dir = 'tmp/poa'+net_name+'_'+str(g_mult)
    fc = tNet.fcoeffs
    fcUC = cars.UC_fcoeffs(fc)
    thetaSO, aSO, rms = cars.get_approx_fun(fcoeffs=fc, nlines=10, range_=[0, 2.7], plot=False)
    thetaUC, aUC, rms = cars.get_approx_fun(fcoeffs=fcUC, nlines=10, range_=[0, 2.7], plot=False)

    gs = np.linspace(0.5, 4, 50)
    for g in gs:
        SO_obj = solve_SO(tNet, g, thetaSO, aSO, MSA=True)
        UE_obj = solve_UE(tNet, g, thetaUC, aUC, MSA=True)
        PoA = UE_obj/SO_obj
        print(PoA)
