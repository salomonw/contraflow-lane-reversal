import src.tnet as tnet
import networkx as nx
import matplotlib.pyplot as plt


def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs


tNet, fcoeffs = read_net('test_9')
#tNet.TAP = tNet.build_TAP(tNet.G)
#tNet.read_node_coordinates(posFile)
#tNet.solveMSA()

#pos = nx.get_node_attributes(tNet.G, 'pos')
#nx.draw(tNet.G, pos)
a = nx.nx_agraph.to_agraph(tNet.G)
a.layout(prog='neato')            # neato layout
a.draw('test3.png' )
print(a)