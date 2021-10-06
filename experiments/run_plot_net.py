import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
plt.style.use(['science','ieee', 'high-vis'])

def read_net(net_name):
    if net_name == 'NYC':
        tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs

# PARAMETERS
# Define net to read
net_name = 'test_9'
g_mult = 1.5
# Read network
tNet, fcoeffs = read_net(net_name)
dir_out = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + 'contraflow'
tNet.build_supergraph(identical_G=True)
# Read position file
tNet.read_node_coordinates('data/pos/' + net_name + '.txt')
# Multiply demand
g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
tNet.set_g(g_per)
# Preprocessing
tNet, max_caps = cflow.integralize_inputs(tNet)
out_dir = 'results/'+dir_out + '_'+ net_name
n_lines_CARS = 5

tNet0 = deepcopy(tNet)
g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
tNet.set_g(g_per)
nominal_capacity = {(i, j): tNet0.G_supergraph[i][j]['capacity'] for i, j in tNet0.G_supergraph.edges()}

tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False,
                                                rebalancing=False, linear=False, bush=True)

edges, weights = zip(*nx.get_edge_attributes(tNet.G_supergraph, 'flow').items())

weights = [tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_k'] for i, j in edges]
min_w = min(weights)
max_w = max(weights)

#tNet0, objfull = cflow.solve_alternating(tNet0, g_per=g_per, e=1e-2, type_='full', n_lines_CARS=n_lines_CARS)
tNet0, objfull, TT, d_norm, RG = cflow.solve_FW(tNet0, step='FW', n_iter=2000)
opt_capacity = {(i, j): tNet0.G_supergraph[i][j]['capacity'] for i, j in tNet0.G_supergraph.edges()}

change_edges = [(i, j) for i, j in tNet.G_supergraph.edges() if (nominal_capacity[(i, j)] < opt_capacity[(i, j)]) or (nominal_capacity[(i, j)] > opt_capacity[(i, j)])]
stay_edges = [(i, j) for i, j in tNet.G_supergraph.edges() if nominal_capacity[(i, j)] == opt_capacity[(i, j)]]
weights = [tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_k'] for i, j in stay_edges]
weights2 = [tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_k'] for i, j in change_edges]

node_demand = [sum(v for k, v in tNet0.g.items() if k[1] == n) for n in tNet0.G.nodes()]
node_size = [x / max(node_demand) * 2 for x in node_demand]


#lanes = nx.get_edge_attributes(tNet0.G_supergraph, 'lanes')
lanes = [nominal_capacity[(u,v)]/1500/2 for u,v in tNet.G.edges]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5*1.5, 2.5*1.5))
cmap = plt.cm.plasma_r
normalize = matplotlib.colors.Normalize(vmin=min_w, vmax=max_w)
tnet.plot_network(tNet.G, ax[0], edgelist=stay_edges, edgecolors=weights, edge_width=lanes, cmap=cmap, nodesize=80, vmin=min_w, vmax=max_w)
tnet.plot_network(tNet.G, ax[0], edgelist=change_edges, edgecolors=weights2, edge_width=lanes, arrowsize=4, cmap=cmap,
                  nodesize=80, vmin=min_w, vmax=max_w, linkstyle='--')

lanes = [opt_capacity[(u,v)]/1500/2 for u,v in tNet.G.edges]
weights3 = [tNet0.G_supergraph[i][j]['flow'] * tNet0.G_supergraph[i][j]['t_k'] for i, j in edges]
tnet.plot_network(tNet0.G, ax[1], edgecolors=weights3, edge_width=lanes, cmap=cmap, nodesize=80, vmin=min_w, vmax=max_w)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
fig.colorbar(sm, ax=ax.ravel().tolist())
plt.savefig('hola.pdf')
