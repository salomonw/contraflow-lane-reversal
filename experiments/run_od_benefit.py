'''
import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc

plt.style.use(['science', 'ieee', 'grid', 'high-vis'])


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
    out_dir = 'results/' + net_name + '_' + str(g_mult)
    tmp_dir = 'tmp/' + net_name + '_' + str(g_mult)
    n_lines_CARS = 11

    theta, a, rms = cars.get_approx_fun(fcoeffs=tNet.fcoeffs, nlines=n_lines_CARS, range_=[0, 2], plot=False)

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
                obj = tnet.get_totalTravelTime(tNet_.G_supergraph, tNet_.fcoeffs)
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
    plt.savefig('lambda.pdf')
    plt.show()

    zdump(objs, out_dir + '/lambda_results.pkl')

'''
















import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import experiments.build_NYC_subway_net as nyc
import networkx as nx

def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs


def run_restricted_model(tNet0, betas, breaks, max_lanes=None, max_links=None):
    tNet = deepcopy(tNet0)
    sol = cflow.solve_opt_int_pwl(tNet,
                                  betas=betas,
                                  breaks=breaks,
                                  max_links=max_links,
                                  max_lanes=max_lanes)
    for i, j in tNet.G.edges():
        tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500
        tNet.G_supergraph[i][j]['t_k'] = cars.travel_time(tNet, i, j)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    return tNet, obj

def plot_od_pair_diff(tNetorg, tNet0):
    tt_org = {}
    tt_new = {}
    tt_imp = {}
    demandVec = []
    ttVec = []
    tt_orgs = {}
    tt_news = {}
    tt_impr = {}
    flowVec = []
    ttVecs = []
    colorsVec = []

    # Compare OD Pairs travel times.
    tt_org = {}
    for od in tNetorg.g:
        tt_org[od] = nx.shortest_path_length(tNetorg.G_supergraph, od[0], od[1], weight='t_k')
    tt_orgs = {}
    for a in tNetorg.G_supergraph.edges():
        tt_orgs[a] = tNetorg.G_supergraph[a[0]][a[1]]['t_k']*tNetorg.G_supergraph[a[0]][a[1]]['flow']
    #'''
    for od in tNet0.g:
        tt_new[od] = nx.shortest_path_length(tNet0.G_supergraph, od[0], od[1], weight='t_k')
        tt_imp[od] = (tt_org[od] - tt_new[od]) / tt_org[od] * 100
        demandVec.append(tNet0.g[od])
        ttVec.append(tt_imp[od])
    #'''
    a0 = [None, None]
    for a in tNet0.G_supergraph.edges():
        tt_news[a] = tNet0.G_supergraph[a[0]][a[1]]['t_k']*tNet0.G_supergraph[a[0]][a[1]]['flow']
        try:
            tt_impr[a] = (tt_orgs[a] - tt_news[a]) / tt_orgs[a] * 100
        except:
            tt_impr[a] = 0
        flowVec.append(tNet0.G_supergraph[a[0]][a[1]]['flow'])
        # if tNet.G_supergraph[a[0]][a[1]]['flow'] > 8000 and tNet.G_supergraph[a[0]][a[1]]['flow'] < 9000:

        if tt_impr[a] > 40:
            colorsVec.append('lime')
            a0 = a
        else:
            colorsVec.append('b')
        ttVecs.append(tt_impr[a])

    idx = 0
    for a in tNet.G_supergraph.edges():
        if a == (a0[1], a0[0]):
            colorsVec[idx] = 'lime'
        idx += 1
    return flowVec, ttVecs, colorsVec, ttVec

def plot_LR(x, y, c=False):
    x = np.array(x)
    y = np.array(y)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='red', linestyle='--')
    plt.scatter(x, y, s=18, alpha=1, c=c)


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
    n_lines_CARS = 11
    theta, a, rms = cars.get_approx_fun(fcoeffs=tNet.fcoeffs, nlines=n_lines_CARS, range_=[0, 2], plot=False)
    # Run experiment
    params = {
        'tnet': tNet,
        'fcoeffs': tNet.fcoeffs,
        'n': n_lines_CARS,
        'theta': theta,
        'a': a,
        'exogenous_G': False,
        'rebalancing': False,
        'linear': False,
        'bush': True,
        'capacity': True,
        'lambda_cap': 1e1,
        'integer': False,
        'max_reversals': 0
    }

    # Get current travel times
    tNet0 = deepcopy(params['tnet'])
    params['capacity'] = False
    params['linear'] = False
    tNet, runtime, od_flows, c = cars.solve_bush_CARSn(**params)
    nom_obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    max_flow = max([tNet.G_supergraph[i][j]['flow'] / tNet.G_supergraph[i][j]['capacity'] for i, j in
                    tNet.G_supergraph.edges()])
    print(max_flow)

    # Optimize using convex relaxation
    params['tnet'] = deepcopy(tNet0)
    params['max_reversals'] = 500000000
    params['capacity'] = True
    params['linear'] = False
    params['integer'] = True
    tNetNew, runtime, od_flows, c = cars.solve_bush_CARSn(**params)
    tNetNew, max_caps = cflow.integralize_inputs(tNetNew)
    max_flow = max([tNetNew.G_supergraph[i][j]['flow'] / tNetNew.G_supergraph[i][j]['capacity'] for i, j in
                    tNetNew.G_supergraph.edges()])
    print(max_flow)
    # Get travel times with convex relaxation solution
    params['tnet'] = deepcopy(tNetNew)
    params['max_reversals'] = 0
    params['capacity'] = False
    params['linear'] = False
    params['integer'] = False
    tNetNew, runtime, od_flows, c = cars.solve_bush_CARSn(**params)
    obj = tnet.get_totalTravelTime(tNetNew.G_supergraph, tNetNew.fcoeffs)

    print(obj/nom_obj*100)

    mkdir_n(out_dir)
    # PLOT RESULTS PER LINK
    fig, ax = plt.subplots(figsize=(4, 2))
    flowVec, ttVecs, colorsVec, ttVec = plot_od_pair_diff(tNet, tNetNew)
    plt.hist(ttVec)
    plt.show()
    plt.axhline(0, linestyle=':', color='k')
    plot_LR(x=flowVec, y=ttVecs, c=colorsVec)
    plt.xlim(0, max(flowVec))
    # plt.ylim(-5, 5)
    plt.xlabel('Flow')
    plt.ylabel('Improvement (\\%)')
    plt.tight_layout()
    plt.savefig(out_dir + '/link_scatter.pdf')
    plt.show()


'''
import src.tnet as tnet
import src.CARS as cars
import src.contraflow as cflow
from src.utils import *
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import experiments.build_NYC_subway_net as nyc
from copy import deepcopy

def read_net(net_name):
    if net_name == 'NYC':
        net, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        net = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs

def more_and_more_lanes(tNetc, max_lanes_vec, gmult=1, n_lines_CARS=5):
    tNet = deepcopy(tNetc)
    objs = []
    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False, rebalancing=False,linear=False, bush=True)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    for m in max_lanes_vec:
        betas = {}
        breaks = {}
        for i, j in tNet.G_supergraph.edges():
            beta0, beta1, breaks0 = cflow.get_arc_pwfunc(tNet, i, j)
            betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}
            breaks[(i, j)] = breaks0
        sol = cflow.solve_opt_int_pwl(tNet, betas=betas, breaks=breaks, max_lanes=m)
        for i, j in tNet.G.edges():
            tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500
            tNetc.G_supergraph[i][j]['t_k'] = cars.travel_time(tNet, i, j)
        obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
        print('max lane changes: ' + str(m)+', obj: ' + str(obj))
        objs.append(obj)
    return objs


if __name__ == "__main__":
    # PARAMETERS
    net_name = str(sys.argv[1])
    g_mult = float(sys.argv[2])
    # Read network
    tNet, fcoeffs = read_net(net_name)
    dir_out = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + 'contraflow'
    tNet.build_supergraph(identical_G=True)
    # Multiply demand
    g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
    tNet.set_g(g_per)
    # Preprocessing
    tNet, max_caps = cflow.integralize_inputs(tNet)
    out_dir = 'tmp/'+net_name+'_'+str(g_mult)
    n_lines_CARS = 5


    tNet0 = copy.deepcopy(tNet)
    tNet0, runtime, od_flows = cars.solve_bush_CARSn(tNet0, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False,
                                                     rebalancing=False, linear=False, bush=True)
    max_lanes_vec = [30]
    tt_org = {}
    tt_new = {}
    tt_imp = {}
    demandVec = []
    ttVec = []
    tt_orgs = {}
    tt_news = {}
    tt_impr = {}
    flowVec = []
    ttVecs = []
    colorsVec = []
    for od in tNet0.g:
        tt_org[od] = nx.shortest_path_length(tNet0.G_supergraph, od[0], od[1], weight='t_k')
    for a in tNet0.G_supergraph.edges():
        tt_orgs[a] = tNet0.G_supergraph[a[0]][a[1]]['t_k']

    objs = more_and_more_lanes(tNet0, max_lanes_vec, gmult=g_mult, n_lines_CARS=n_lines_CARS)

    for od in tNet0.g:
        tt_new[od] = nx.shortest_path_length(tNet0.G_supergraph, od[0], od[1], weight='t_k')
        tt_imp[od] = (tt_org[od ] -tt_new[od] ) /tt_org[od ] *100
        demandVec.append(tNet0.g[od])
        ttVec.append(tt_imp[od])
    for a in tNet0.G_supergraph.edges():
        tt_news[a] = tNet0.G_supergraph[a[0]][a[1]]['t_k']
        tt_impr[a] = (tt_orgs[a ] -tt_news[a] ) /tt_orgs[a ] *100
        flowVec.append(tNet0.G_supergraph[a[0]][a[1]]['flow'])
        # if tNet.G_supergraph[a[0]][a[1]]['flow'] > 8000 and tNet.G_supergraph[a[0]][a[1]]['flow'] < 9000:
        if tt_impr[a ] >40:
            colorsVec.append('lime')
            a0 = a
        else:
            colorsVec.append('b')
        ttVecs.append(tt_impr[a])

    idx = 0
    for a in tNet.G_supergraph.edges():
        if a == (a0[1], a0[0]):
            colorsVec[idx] = 'lime'
        idx += 1


    def plot_LR(x ,y, c=False):
        x = np.array(x)
        y = np.array(y)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m* x + b, color='red', linestyle='--')
        plt.scatter(x, y, s=18, alpha=1, c=c)


    mkdir_n(out_dir)

    # PLOT RESULTS PER LINK
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.axhline(0, linestyle=':', color='k')
    plot_LR(x=flowVec, y=ttVecs, c=colorsVec)
    plt.xlim(0, max(flowVec))
    # plt.ylim(-5, 5)
    plt.xlabel('Flow')
    plt.ylabel('Improvement (\\%)')
    plt.tight_layout()
    plt.savefig(out_dir + '/link_scatter.pdf')
    # plt.show()

    plt.cla()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.hist(ttVecs, bins=20)
    plt.xlabel('Improvement (\\%)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(out_dir + '/link_hist.pdf')
    # plt.show()

    
    # PLOT RESULTS PER OD PAIR
    plt.cla()
    fig, ax = plt.subplots(figsize=(4,2))
    plt.axhline(0, linestyle=':', color='k')
    plot_LR(x=demandVec, y=ttVec)
    plt.xlim(0, max(demandVec))
    plt.xlabel('Demand')
    plt.ylabel('Improvement (\\%)')
    plt.tight_layout()
    plt.savefig(out_dir + '/OD_scatter.pdf')
    
    plt.cla()
    fig, ax = plt.subplots(figsize=(4,2))
    plt.hist([v for k,v in tt_imp.items()], bins=40, range=(-10,15))
    plt.xlabel('Improvement (\\%)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(out_dir + '/OD_hist.pdf')
    #plt.show()
    
    zdump(tt_org, out_dir + '/travel_time_original.pkl')
    zdump(tt_new, out_dir + '/travel_time_new.pkl')
    zdump(tt_imp, out_dir + '/travel_time_improvements.pkl')
'''