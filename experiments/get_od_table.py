import src.tnet as tnet
import pandas as pd

net_name = 'EMA'
delta = 0.001
g_mult = 1

# Read network
netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name)
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)

# Get dataframe
print(tNet.g)
#tnet.plot_network(tNet.G)
df = pd.DataFrame(tNet.g)


# Print latex table
print(df.to_latex())