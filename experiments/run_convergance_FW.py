import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc
plt.style.use(['science','ieee','high-vis'])

def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs


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
    out_dir = 'results/'+net_name+'_'+str(g_mult)
    tmp_dir = 'tmp/'+net_name+'_'+str(g_mult)
    n_lines_CARS = 11
    theta, a, rms = cars.get_approx_fun(fcoeffs=tNet.fcoeffs, nlines=n_lines_CARS, range_=[0, 2], plot=False)


    objs = {}
    tNet0 = deepcopy(tNet)
    mkdir_n(out_dir)
    mkdir_n(tmp_dir)

    g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
    tNet.set_g(g_per)


    params = {
        'tnet': tNet0,
        'fcoeffs': tNet0.fcoeffs,
        'n': n_lines_CARS,
        'theta': theta,
        'a': a,
        'exogenous_G': False,
        'rebalancing': False,
        'linear': False,
        'bush': True,
        'capacity': True,
        'lambda_cap': 1e4,
        'integer': True,
        'max_reversals': 9999
    }
    n_iter = 1000


    tNet_ = deepcopy(tNet0)
    params_ = deepcopy(params)
    params_['tnet'] = tNet_

    tNet_, obj, TT, dnorm, RG, c, runtime = cflow.solve_FW(tNet_, step='FW', n_iter=n_iter)

    fig, ax = plt.subplots(3, figsize=(4,3), sharex=True)

    ax[0].plot(TT)
    ax[1].plot(dnorm)
    ax[2].plot(RG)
    ax[0].set_xlim((0, n_iter))
    plt.legend()
    ax[1].set_xlim((0, n_iter))
    ax[2].set_xlim((0, n_iter))
    ax[0].set_ylabel('$J(\\mathbf{x}, \\mathbf{z})$')
    ax[1].set_ylabel('$\\|\\nabla J(\mathbf{z}) \\|_2$')
    ax[2].set_ylabel('RG')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[2].set_xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(out_dir + '/FW_iteration_mult' + str(g_mult) + '.pdf')

