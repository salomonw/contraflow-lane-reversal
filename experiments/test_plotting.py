import src.tnet as tnet
import networkx as nx
import matplotlib.pyplot as plt

netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters('Braess1')
posFile = "data/pos/Braess1_pos.txt"
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
#tNet.TAP = tNet.build_TAP(tNet.G)
tNet.read_node_coordinates(posFile)
tNet.solveMSA()

pos = nx.get_node_attributes(tNet.G, 'pos')
nx.draw(tNet.G, pos)
plt.show()

tNet.build_supergraph()
