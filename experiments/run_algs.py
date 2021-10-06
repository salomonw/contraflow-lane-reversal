import src.tnet as tnet
import copy
import src.CARS as cars
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#import src.pwapprox as pw
from gurobipy import *
import pwlf as pw
from src.utils import *
from datetime import datetime
import experiments.build_NYC_subway_net as nyc
import itertools

plt.style.use(['science','ieee', 'high-vis'])


def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return tNet, fcoeffs


if __name__ == "__main__":
    # PARAMETERS
    net_name = str(sys.argv[1])
    g_mult = float(sys.argv[2])

    # Read network
    tNet, fcoeffs = read_net(net_name)
    dir_out = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + 'contraflow'

    # multiply demand
    g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
    tNet.set_g(g_per)

    tNet.build_supergraph(identical_G=True)

    # PREPROCESSING!
    for i,j in tNet.G_supergraph.edges():
        try: tNet.G_supergraph[j][i]
        except:
            tNet.G_supergraph.add_edge(j, i, capacity=0, t_0=tNet.G_supergraph[i][j]['t_0'], length=tNet.G_supergraph[i][j]['length'])
    maxcaps = {(i,j) : tNet.G_supergraph[i][j]['capacity']+tNet.G_supergraph[j][i]['capacity'] for i,j in tNet.G_supergraph.edges()}
    nx.set_edge_attributes(tNet.G_supergraph, 0, 'lanes')
    nx.set_edge_attributes(tNet.G_supergraph, 0, 'max_lanes')
    nx.set_edge_attributes(tNet.G_supergraph, 0, 'max_capacity')
    for i,j in tNet.G_supergraph.edges():
        tNet.G_supergraph[i][j]['lanes'] = max(np.round(tNet.G_supergraph[i][j]['capacity']/1500), 1)
        if tNet.G_supergraph[j][i] == False:
            print('a')
    for i,j in tNet.G.edges():
        tNet.G_supergraph[i][j]['capacity'] = tNet.G_supergraph[i][j]['lanes']*1500
        tNet.G_supergraph[i][j]['max_capacity'] = (tNet.G_supergraph[i][j]['lanes'] + tNet.G_supergraph[j][i]['lanes'])*1500
        tNet.G_supergraph[i][j]['max_lanes'] = tNet.G_supergraph[i][j]['lanes'] + tNet.G_supergraph[j][i]['lanes']



    # PREPROCESSING!
    plt.figure()
    TT, runtime = tNet.solveMSAsocial_supergraph(build_t0=False, exogenous_G=False)
    plt.plot(list(range(len(TT))), TT, label='Fixed')
    TT, runtime = tNet.solveMSAsocial_capacity_supergraph(build_t0=False, exogenous_G=False)
    plt.plot(list(range(len(TT))), TT, label='FW')

    plt.show()