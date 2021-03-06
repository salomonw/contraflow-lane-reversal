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

def plot_alternating(ax, obj1b1, obj5,  objfull, nom_obj):

    obj1b1 = [i/nom_obj*100 for i in obj1b1]
    obj5 = [i/nom_obj*100 for i in obj5]
    objfull = [i/nom_obj*100 for i in objfull]

    objs_labels = []
    objs_labels.append('One')
    objs_labels.append('Five')
    objs_labels.append('All')
    objs[g_mult].append(obj1b1[-1])
    objs[g_mult].append(obj5[-1])
    objs[g_mult].append(objfull[-1])

    #zdump(obj1b1, tmp_dir + '/iteration_one_mult_' + str(g_mult) + '.pkl')
    #zdump(obj5, tmp_dir + '/iteration_five_mult_' + str(g_mult) + '.pkl')
    #zdump(objfull, tmp_dir + '/iteration_all_mult_' + str(g_mult) + '.pkl')

    ax.plot(obj1b1, label='One', marker='o')
    ax.plot(obj5, label='Five', marker='o')
    ax.plot(objfull, label='All', marker='o')
    ax.set_xlim((0, len(obj5)-1))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Obj/Nominal Obj (\%)')

    plt.tight_layout()
    plt.legend(frameon=1, facecolor='white', framealpha=0.9, loc=1)

    plt.savefig(out_dir + '/sequential_iteration_mult' + str(g_mult) + '.pdf')
    #plt.show()



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

    #fig, ax = plt.subplots(figsize=(4,2))
    theta, a, rms = cars.get_approx_fun(fcoeffs=tNet.fcoeffs, nlines=n_lines_CARS, range_=[0, 3], plot=False)
    #plt.tight_layout()
    #plt.legend(frameon=1, facecolor='white', framealpha=0.9, loc=2)
    #plt.savefig('approx.pdf')
    #plt.show()

    objs = {}
    tNet0 = deepcopy(tNet)
    mkdir_n(out_dir)
    mkdir_n(tmp_dir)
    print('--------------------------------------------------------')
    print('Method \t Obj (%) \t Run time (ms) \t Max Flow \t Total Cap')
    print('--------------------------------------------------------')


    for g_mult in [g_mult]:
        g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
        tNet.set_g(g_per)
        algs = ['Nom', 'FW', 'Alt1', 'Alt5', 'AltF', 'MIQP', 'QP', 'MILP', 'LP']
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
            'lambda_cap': 1e4,
            'integer': True,
            'max_reversals': 9999
        }

        for alg in algs:
            tNet_ = deepcopy(tNet0)
            params_ = deepcopy(params)
            params_['tnet'] = tNet_
            if alg == 'Nom':
                params_['capacity'] = False
            elif alg == 'MIQP':
                params_ = params_
            elif alg == 'QP':
                params_['integer'] = False
            elif alg == 'MILP':
                params_['linear'] = True
            elif alg == 'LP':
                params_['linear'] = True
                params_['integer'] = False

            if alg in ['Nom', 'MIQP', 'QP', 'MILP', 'LP']:
                tNet_, runtime, od_flows, c = cars.solve_bush_CARSn(**params_)

            if alg =='FW':
                tNet_, obj, TT, dnorm, RG, c, runtime = cflow.solve_FW(tNet_, step=alg, n_iter=4000)
                for (i,j), cap in c.items():
                    tNet_.G_supergraph[i][j]['capacity'] = cap
                    tNet_.G_supergraph[i][j]['lanes'] = cap/1500

            elif alg =='Alt1':
                tNet_, obj1b1, runtime = cflow.solve_alternating(tNet_, e=1e-2, type_='one by one',
                                                             n_lines_CARS=n_lines_CARS, theta=theta, a=a)
            elif alg =='Alt5':
                tNet_, obj5, runtime = cflow.solve_alternating(tNet_, e=1e-2, type_=5,
                                                             n_lines_CARS=n_lines_CARS, theta=theta, a=a)
            elif alg =='AltF':
                tNet_, objfull, runtime = cflow.solve_alternating(tNet_, e=1e-2, type_='full',
                                                              n_lines_CARS=n_lines_CARS, theta=theta, a=a)

            # Evaluation
            params_ = deepcopy(params)
            params_['capacity'] = False
            params_['linear'] = False
            tNet_ = cflow.project_fluid_sol_to_integer(tNet_)
            params_['tnet'] = tNet_
            tNet_, _, od_flows, c = cars.solve_bush_CARSn(**params_)
            tot_cap = sum(tNet_.G_supergraph[i][j]['capacity'] for i,j in tNet_.G_supergraph.edges())
            max_flow = max([tNet_.G_supergraph[i][j]['flow'] / tNet_.G_supergraph[i][j]['capacity'] for i, j in tNet_.G_supergraph.edges()])
            obj = tnet.get_totalTravelTime(tNet_.G_supergraph, tNet_.fcoeffs)
            if alg == 'Nom':
                nom_obj = obj

            print('{} \t {:.1f} \t {:.2f} \t {:.1f}'.format(alg, 100-obj / nom_obj * 100, runtime, max_flow))

    fig, ax1 = plt.subplots(figsize=(4,2))
    plot_alternating(ax1, obj1b1, obj5, objfull, nom_obj)

    print('--------------------------------------------------------')
    zdump(objs, out_dir + '/obj_results.pkl')


    '''
    steps = ['FW']#, 'MSA', 5e-1, 5e0, 5e1]
    n_iter = 4000
    for step in steps:
        tNet_ = deepcopy(tNet0)
        tNet_.set_g(g_per)
        tNet_, obj, TT, dnorm, RG, c, runtime = cflow.solve_FW(tNet_, step, n_iter=n_iter)
        print('FW ('+str(step)+') \t {:.1f} \t {:.1f} '.format(obj/nom_obj*100, runtime*1000))
        objs_labels.append('FW: ' + str(step))
        objs[g_mult].append(obj)
        zdump(TT, tmp_dir + '/step_' + str(step) + '_TT_' + str(g_mult) + '.pkl')
        zdump(dnorm, tmp_dir + '/step_' + str(step) + '_dnorm_' + str(g_mult) + '.pkl')
        zdump(RG, tmp_dir + '/step_' + str(step) + '_RG_' + str(g_mult) + '.pkl')
    #ax[0].set_xlim((0, n_iter))
    #plt.legend()
    #ax[1].set_xlim((0, n_iter))
    #ax[2].set_xlim((0, n_iter))
    #ax[0].set_xlabel('Iteration')
    #ax[0].set_ylabel('Objective')
    #ax[0].set_yscale('log')
    #ax[0].set_yscale('log')
    #ax[1].set_xlabel('Iteration')
    #ax[1].set_ylabel('Derivative norm')
    #ax[1].set_yscale('log')
    #ax[2].set_xlabel('Iteration')
    #ax[2].set_ylabel('Relative Gap')
    #ax[2].set_yscale('log')

    #plt.tight_layout()
    #plt.savefig(out_dir + '/FW_iteration_mult' + str(g_mult) + '.pdf')
    '''
