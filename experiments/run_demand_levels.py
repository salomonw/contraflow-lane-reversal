import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc


def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs

def comparison_ILP_fluid(tNet):
    tNeta = deepcopy(tNet)
    tNetc = deepcopy(tNet)
    # solve original
    obj0 = tnet.get_totalTravelTime(tNetc.G_supergraph, tNetc.fcoeffs)
    print('Solve Nominal!')
    # solve fluidic
    objf, psi_v, ns, x, caps, tNetc = cflow.solve_opt_fluid(tNetc, sequential=False, psi=9999, eps=1e-10,)
    objfluid = tnet.get_totalTravelTime(tNetc.G_supergraph, fcoeffs=tNetc.fcoeffs)
    print('Solve Fluidic!')
    # easy map fluidic to integer
    cflow.project_fluid_sol_to_integer(tNetc)
    objfluidInt = tnet.get_totalTravelTime(tNetc.G_supergraph, tNetc.fcoeffs)
    print('Solve Fluidic Projection!')
    # solve optimal ILP problem
    cflow.solve_optimal_ILP(tNeta)
    objILP = tnet.get_totalTravelTime(tNeta.G_supergraph, tNeta.fcoeffs)
    print('Solve ILP Projection!')
    return obj0, objILP, objfluid, objfluidInt

def compare_diff_g(tNet0, gs, n_lines_CARS):
    obj0v = []
    objILPv = []
    objfluidv = []
    objfluidIntv = []
    for g in gs:
        tNet = deepcopy(tNet0)
        g_per = tnet.perturbDemandConstant(tNet.g, g)
        tNet.set_g(g_per)
        #tNet.build_supergraph(identical_G=True)
        #tNet.solveMSA_supergraph()
        tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet,
                                                        fcoeffs=tNet.fcoeffs,
                                                        n=n_lines_CARS,
                                                        exogenous_G=False,
                                                        rebalancing=False,
                                                        linear=False,
                                                        bush=True)
        print('Solved CARS!')
        obj0, objILP, objfluid, objfluidInt = comparison_ILP_fluid(tNet)
        #print([obj0, objILP, objfluid, objfluidInt])
        obj0v.append((obj0/objILP-1)*100)
        objILPv.append((objILP/objILP-1)*100)
        objfluidv.append((objfluid/objILP-1)*100)
        objfluidIntv.append((objfluidInt/objILP-1)*100)
    return obj0v, objILPv, objfluidv, objfluidIntv

def plot_diff_gs(gs, obj0, objILP, objfluid, objfluidInt):
    fig, ax = plt.subplots()
    ax.plot(gs, objILP, label='ILP')#, linestyle=':', color='red',  label='ILP')
    ax.plot(gs, objfluid, label='Relaxation', marker='.')#, linestyle='-', color='blue', label='Relaxation')
    ax.plot(gs, objfluidInt, label='Relaxation$\\rightarrow$Proj', marker='o')#, linestyle='-', color='blue', label='Relaxation')
    ax.plot(gs, obj0, label='Original', marker='+')#, linestyle='--', color='green', label='Actual')
    plt.xlabel('Demand multiplier')
    plt.ylabel('Performance (\\%)')
    plt.xlim((min(gs), max(gs)))
    plt.tight_layout()
    plt.legend()
    return fig, ax


if __name__ == "__main__":
    # PARAMETERS
    net_name = str(sys.argv[1])
    g_mult = float(sys.argv[2])

    # Read network
    tNet, fcoeffs = read_net(net_name)
    tNet.build_supergraph(identical_G=True)
    # Multiply demand
    g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
    tNet.set_g(g_per)
    # Preprocessing
    tNet, max_caps = cflow.integralize_inputs(tNet)
    out_dir = 'tmp/'+net_name+'_'+str(g_mult)
    n_lines_CARS = 5


    tNet0 = deepcopy(tNet)
    gs = [1, 1.5, 2, 2.5, 3]
    objs = compare_diff_g(tNet0, gs, n_lines_CARS)
    obj0, objILP, objfluid, objfluidInt = objs
    print(' Alg. \t Obj \n Nominal \t {} \n ILP \t {} \n Fluid \t {} \n Fluid Proj. \t {}'.format(
        round(obj0[0], 2), round(objILP[0],2), round(objfluid[0],2), round(objfluidInt[0], 2)
    ))
    mkdir_n(out_dir)
    zdump(objs, out_dir + '/objs_demands.pkl')