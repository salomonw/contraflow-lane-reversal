import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc
plt.style.use(['science','ieee', 'high-vis'])

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
    params['capacity'] = False
    params['max_reversals'] = 0
    params['lambda_cap'] = 0
    #print(params)
    tNet, runtime, _ = cars.solve_bush_CARSn(tNet, **params)
    #print({(i,j): tNet.G_supergraph[i][j]['flow'] for i, j in tNet.G_supergraph.edges()})
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
    out_dir = 'results/'+net_name+'_'+str(g_mult)
    tmp_dir = 'tmp/'+net_name+'_'+str(g_mult)
    n_lines_CARS = 9

    objs = {}
    tNet0 = deepcopy(tNet)
    mkdir_n(out_dir)
    mkdir_n(tmp_dir)
    print('\n \n \t Method \t Obj \t Time')
    for g_mult in [g_mult]:
        g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
        tNet.set_g(g_per)
        # print(min([tNet.G_supergraph[i][j]['max_capacity'] for i,j in tNet.G_supergraph.edges()]))
        objs[g_mult] = []
        objs_labels = ['Nominal']
        params = {
            'fcoeffs' :tNet.fcoeffs,
            'n': n_lines_CARS,
            'exogenous_G': False,
            'rebalancing': False,
            'linear': False,
            'bush': True,
            'od_flows_flag': False
        }
        tNet, m, od_flows = cars.solve_bush_CARSn(tNet, **params)
        obj0 = eval_obj(tNet, params)
        #obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
        objs[g_mult].append(obj0)
        print('\t Nominal \t {:.3f} \t{:.3f}'.format(obj0, m.Runtime))

        steps = ['FW']#, 'MSA', 5e-1, 5e0, 5e1]
        n_iter = 8000
        for step in steps:
            tNet_ = deepcopy(tNet0)
            tNet_.set_g(g_per)
            tNet_, obj, TT, dnorm, RG, c, runtime = cflow.solve_FW(tNet_, step, n_iter=n_iter)
            obj = eval_obj(tNet_, params)
            #print('FW ( ' +str(step) +') : ' + str(obj))
            objs_labels.append('FW: ' + str(step))
            objs[g_mult].append(obj)
            zdump(TT, tmp_dir + '/step_' + str(step) + '_TT_' + str(g_mult) + '.pkl')
            zdump(dnorm, tmp_dir + '/step_' + str(step) + '_dnorm_' + str(g_mult) + '.pkl')
            zdump(RG, tmp_dir + '/step_' + str(step) + '_RG_' + str(g_mult) + '.pkl')
            print('\t {} \t {:.3f} \t{:.3f}'.format(step, 1-obj/obj0, runtime))
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

        fig, ax = plt.subplots()
        tNet_, obj1b1, runtime = cflow.solve_alternating(tNet0, g_per=g_per, e=1e-2, type_='one by one', n_lines_CARS=n_lines_CARS)
        objb1 = eval_obj(tNet_, params)
        print('\t Alt.(1) \t {:.3f} \t{:.3f}'.format(1-objb1/obj0, runtime))
        #print('one: ' + str(objb1))
        tNet_, obj5, runtime = cflow.solve_alternating(tNet0, g_per=g_per, e=1e-2, type_=5, n_lines_CARS=n_lines_CARS)
        objb5 = eval_obj(tNet_, params)
        print('\t Alt.(5) \t {:.3f} \t{:.3f}'.format(1-objb5/obj0, runtime))
        #print('five: ' + str(objb5))
        tNet_, objfull, runtime = cflow.solve_alternating(tNet0, g_per=g_per, e=1e-2, type_='full', n_lines_CARS=n_lines_CARS)
        objbfull = eval_obj(tNet_, params)
        print('\t Alt.(full) \t {:.3f} \t{:.3f}'.format(1-objbfull/obj0, runtime))
        #print('all: ' + str(objbfull))

        objs_labels.append('One')
        objs_labels.append('Five')
        objs_labels.append('All')
        objs[g_mult].append(objb1)
        objs[g_mult].append(objb5)
        objs[g_mult].append(objbfull)

        zdump(obj1b1, tmp_dir + '/iteration_one_mult_' + str(g_mult) + '.pkl')
        zdump(obj5, tmp_dir + '/iteration_five_mult_' + str(g_mult) + '.pkl')
        zdump(objfull, tmp_dir + '/iteration_all_mult_' + str(g_mult) + '.pkl')

        ax.plot(obj1b1, label='One', marker='o')
        ax.plot(obj5, label='Five', marker='o')
        ax.plot(objfull, label='All', marker='o')
        ax.set_xlim((0, len(obj5)-1))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')

        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir + '/sequential_iteration_mult' + str(g_mult) + '.pdf')
        print(objs)

    zdump(objs, out_dir + '/obj_results.pkl')