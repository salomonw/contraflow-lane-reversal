import src.tnet as tnet
import pandas as pd
import numpy as np

net_name = 'test_9'

# Read network
netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name)
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)

nodes = [i+1 for i in range(tNet.nNodes)]
M = np.zeros((tNet.nNodes, tNet.nNodes))
for (i,j), v in tNet.g.items():
    M[i-1, j-1] = v

df = pd.DataFrame(M)

# Print latex table
print(df.to_latex())