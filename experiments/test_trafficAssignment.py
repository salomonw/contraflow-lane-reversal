import src.tnet as tnet

netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters('EMA')
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
tNet.TAP = tNet.build_TAP(tNet.G)
tNet.solveMSA()
print([(i,j, tNet.G[i][j]['flow']) for i,j in tNet.G.edges()])
