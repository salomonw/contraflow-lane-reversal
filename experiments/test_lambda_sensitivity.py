import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc
import matplotlib.font_manager

plt.style.use(['science', 'ieee', 'grid', 'high-vis'])


def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs


def eval_obj(tNet2, params2):
    tNet = deepcopy(tNet2)
    params = deepcopy(params2)
    params['tnet'] = tNet
    params['capacity'] = False
    params['max_reversals'] = 0
    params['lambda_cap'] = 0
    tNet, runtime, _, c = cars.solve_bush_CARSn(**params)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    return obj


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
    out_dir = 'results/' + net_name + '_' + str(g_mult)
    tmp_dir = 'tmp/' + net_name + '_' + str(g_mult)
    n_lines_CARS = 9

    theta, a, rms = cars.get_approx_fun(fcoeffs=tNet.fcoeffs, nlines=n_lines_CARS, range_=[0, 3], plot=False)

    objs = {}
    tNet0 = deepcopy(tNet)
    mkdir_n(out_dir)

    objs_lambda = {}
    for g_mult in [g_mult]:
        g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
        tNet.set_g(g_per)
        algs = ['Nom', 'MIQP', 'QP', 'MILP', 'LP']
        objs[g_mult] = []

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
            'lambda_cap': 1e-9,
            'integer': True,
            'max_reversals': 9999
        }


        for alg in algs:
            L = np.logspace(0.0001, 5, num=10)
            mkdir_n(tmp_dir)
            print('--------------------------------------------------------')
            print('Method \t Lambda \t Obj (%)')
            print('--------------------------------------------------------')
            objs_lambda[alg] = []
            for l in L:
                tNet_ = deepcopy(tNet0)
                params_ = deepcopy(params)
                params_['tnet'] = tNet_
                l = l / 10
                params_['lambda_cap'] = l
                if alg == 'Nom':
                    params_['capacity'] = False
                elif alg == 'MIQP':
                    params_ = params_
                elif alg == 'QP':
                    params_['integer'] = False
                elif alg == 'MILP':
                    params_['linear'] = True
                    params_['integer'] = True
                elif alg == 'LP':
                    params_['linear'] = True
                    params_['integer'] = False

                tNet_, runtime, od_flows, c = cars.solve_bush_CARSn(**params_)
                tNet_ = cflow.project_fluid_sol_to_integer(tNet_)

                params_ = deepcopy(params)
                params_['capacity'] = False
                params_['linear'] = False
                params_['tnet'] = tNet_
                tNet_, runtime, od_flows, c = cars.solve_bush_CARSn(**params_)
                tot_cap = sum(tNet_.G_supergraph[i][j]['capacity'] for i, j in tNet_.G_supergraph.edges())
                max_flow = max([tNet_.G_supergraph[i][j]['flow'] / tNet_.G_supergraph[i][j]['capacity'] for i, j in
                                tNet_.G_supergraph.edges()])
                obj = eval_obj(tNet_, params)
                if alg == 'Nom':
                    nom_obj = obj
                    break

                objs_lambda[alg].append(obj / nom_obj * 100)
                print('{} \t {:.1f} \t {:.1f}'.format(alg, l, obj / nom_obj * 100))
            print('--------------------------------------------------------')

    fig, ax = plt.subplots(figsize=(4,2))
    for k,v in objs_lambda.items():
        if k != 'Nom':
            ax.plot(L, v, label=k)
    #ax.grid()
    ax.set_xlim(min(L), max(L))
    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('Obj/Nominal (\%)')
    plt.legend(facecolor='white', framealpha=0.9, loc=1)
    plt.tight_layout()
    plt.savefig('lambda_'+net_name+'.pdf')
    plt.show()

    zdump(objs, out_dir + '/lambda_results.pkl')



