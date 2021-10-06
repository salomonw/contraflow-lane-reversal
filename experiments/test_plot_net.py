import networkx as nx

import src.tnet as tnet
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc

def read_net(net_name):
    if net_name == 'NYC':
        tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return tNet, fcoeffs

net = 'test_9'
if net == 'NYC':
    fig_size = (6,16)
elif net =='test_9':
    fig_size = (6, 3)
else:
    fig_size = (6, 6)

tNet, fcoeffs = read_net(net)
tNet.read_node_coordinates('data/pos/'+net+'.txt')
tNet.build_supergraph()
tNet.latlong2xy()

fig, ax = plt.subplots()
tnet.plot_network(tNet.G, ax=ax)
if net == 'test_9':
    nx.draw_networkx_nodes(tNet.G,
                           pos=nx.get_node_attributes(tNet.G, 'pos'),
                           alpha=1,
                           node_color='w',
                           ax=ax,
                           edgecolors='k',
                           node_size=300)


ax.figure.set_size_inches(fig_size[0], fig_size[1])
#plt.show()
plt.savefig(net+'.pdf')