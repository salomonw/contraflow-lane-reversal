import src.tnet as tnet
#import matplotlib.pyplot as plt
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
tNet, fcoeffs = read_net(net)
tNet.OD_to_tex()
