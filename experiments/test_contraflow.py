import src.tnet as tnet
import copy
import src.CARS as cars
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
#import src.pwapprox as pw
from gurobipy import *
import pwlf as pw
from src.utils import *
from datetime import datetime
import experiments.build_NYC_subway_net as nyc
import itertools
from multiprocessing import Pool
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.style.use(['science','ieee', 'high-vis'])


def read_net(net_name):
    if net_name == 'NYC':
        tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return tNet, fcoeffs


def get_derivative_ij(args):
    xij, xji, t0ij, t0ji, mij, mji, fcoeffs, delta = args
    t0 = eval_tt_funct(xij, t0ij, mij, fcoeffs)
    tinv0 = eval_tt_funct(xji, t0ji, mji, fcoeffs)
    mij += delta
    mji -= delta
    t = eval_tt_funct(xij, t0ij, mij, fcoeffs)
    tinv = eval_tt_funct(xji, t0ji, mji, fcoeffs)
    v = ((t + tinv) - (t0 + tinv0)) / (delta)
    return v

@timeit
def get_derivative(G, fcoeffs, delta, pool):
    # Find derivatives
    # Case 1: Find derivative estimates by assuming fixed routes
    d = {}
    args = [(G[i][j]['flow'], G[j][i]['flow'], G[i][j]['t_0'], G[j][i]['t_0'], G[i][j]['capacity'], G[j][i]['capacity'], fcoeffs, delta) for i,j in G.edges()]
    results = pool.map(get_derivative_ij, args)
    #print(results)
    k=0
    for i,j in G.edges():
        d[(i,j)] = results[k]
        k+=1
    #manager = Manager()
    #d = manager.dict()
    #job = [Process(target=get_derivative_ij, args=(G[i][j]['flow'], G[j][i]['flow'], G[i][j]['t_0'], G[j][i]['t_0'], G[i][j]['capacity'], G[j][i]['capacity'], fcoeffs, delta)) for i,j in G.edges()]
    #_ = [p.start() for p in job]
    #_ = [p.join() for p in job]
    #print(d)

    #for i,j in G.edges():
    #    v = get_derivative_ij(G[i][j]['flow'], G[j][i]['flow'], G[i][j]['t_0'], G[j][i]['t_0'], G[i][j]['capacity'], G[j][i]['capacity'], fcoeffs, delta)
    #    d[(i,j)] = v


    if len(d)>0:
        a = max(d, key=d.get)
        return d, a
    else:
        return {}, (0,0)

@timeit
def move_capacity(G, d, gamma, gamma0):
    V = []
    for (i, j), g in d.items():
        v = gamma[(i, j)] * g
        m = 1500
        if (i, j) not in V:
            max_c = G[i][j]['max_capacity']
            if G[i][j]['capacity'] - v > max_c - m and G[j][i]['capacity'] + v <= m:
                G[i][j]['capacity'] = max_c - m
                G[j][i]['capacity'] = m
            elif G[i][j]['capacity'] - v <= m and G[j][i]['capacity'] + v > max_c - m:
                G[i][j]['capacity'] = m
                G[j][i]['capacity'] = max_c - m
            else:
                G[i][j]['capacity'] -= v
                G[j][i]['capacity'] += v
            V.append((j, i))
        # k[(i, j)] += 1
        # k+=1
        gamma[(i, j)] = gamma0[(i, j)]  # * 1/np.sqrt(np.sqrt(k)) #*np.exp(-decay*k[(i,j)]/10)  #/(k[(i,j)]**(1/4))
        # print(gamma[(i, j)])
        # gamma = gamma0/k
    return G, v

def solve_opt_fluid(tNet, sequential=False, psi = 9999, eps = 1e-10):
    #gamma0 = {(i, j): 0.9999 for i, j in tNet.G.edges()}
    gamma0 = {(i, j): 10 for i, j in tNet.G_supergraph.edges()}
    gamma = copy.deepcopy(gamma0)
    #k = {(i, j): 1 for i, j in tNet.G.edges()}
    k=0
    caps = {(i, j): [] for i, j in tNet.G_supergraph.edges()}

    obj = []
    psi_v = []
    ns = []
    x = []
    cnt = 0
    delta = 1
    decay = 0.001
    pool = mp.Pool(mp.cpu_count() - 1)
    while psi >= eps and cnt < 5000:
        if sequential:
            #tNet.solveMSA()
            tNet.tNet.solveMSAsocial_supergraph()
            #tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=8, exogenous_G=False, rebalancing=False, linear=False, bush=True)
        d, b = get_derivative(tNet.G_supergraph, fcoeffs, delta, pool)
        tNet.G_supergraph, v = move_capacity(tNet.G_supergraph, d, gamma, gamma0)

        psi = sum([i ** 2 for i in d.values()])
        obj_ = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        obj.append(obj_)
        psi_v.append(psi)
        ns.append(len(d))
        print('i: ' + str(cnt) + ' n: ' + str(len(d)) + ', psi: ' + str(psi) + ', obj: ' + str(obj_))
        x.append(cnt)
        cnt += 1
    pool.close()
    [caps[(i, j)].append(tNet.G_supergraph[i][j]['capacity']) for (i, j) in tNet.G_supergraph.edges()]
    #print('i: ' + str(cnt) + ' n: ' + str(len(d)) + ', psi: ' + str(psi) + ', obj: ' + str(obj_))
    return obj, psi_v, ns, x, caps, tNet

def solve_opt_integer(tNet, sequential=False, psi = 9999, eps = 1e-10):
    gamma0 = {(i, j): .5 for i, j in tNet.G.edges()}
    gamma = gamma0
    k = {(i, j): 1 for i, j in tNet.G.edges()}
    caps = {(i, j): [] for i, j in tNet.G.edges()}

    obj = []
    psi_v = []
    ns = []
    x = []
    cnt = 0
    decay = 0.01
    pool = mp.Pool(mp.cpu_count() - 1)
    while psi >= eps and cnt < 15000:
        if sequential:
            #tNet.solveMSA()
            tNet.solveMSA_supergraph()
            #tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=8, exogenous_G=False, rebalancing=False, linear=False, bush=True)
        d, b = get_derivative(tNet.G_supergraph, tNet.fcoeffs, delta, pool)
        for (i, j), v in d.items():
            if tNet.G_supergraph[i][j]['capacity'] - gamma[(i, j)] * v >= maxcaps[(i, j)] or tNet.G_supergraph[j][i]['capacity'] + gamma[(i, j)] <= 0:
                g = min(maxcaps[(i, j)] - tNet.G_supergraph[i][j]['capacity'], tNet.G_supergraph[j][i]['capacity'])
                tNet.G_supergraph[i][j]['capacity'] -= g + 0.001
                tNet.G_supergraph[j][i]['capacity'] += g - 0.001
            if tNet.G_supergraph[i][j]['capacity'] - gamma[(i, j)] * v <= 0 or tNet.G_supergraph[j][i]['capacity'] + \
                    gamma[(i, j)] >= maxcaps[(i, j)]:
                g = min(maxcaps[(i, j)] - tNet.G_supergraph[j][i]['capacity'], tNet.G_supergraph[i][j]['capacity'])
                tNet.G_supergraph[i][j]['capacity'] -= g + 0.001
                tNet.G_supergraph[j][i]['capacity'] += g - 0.001
            else:
                tNet.G_supergraph[i][j]['capacity'] -= gamma[(i, j)] * v
                tNet.G_supergraph[j][i]['capacity'] += gamma[(i, j)] * v
            k[(i, j)] += 1
            gamma[(i, j)] = gamma0[(i, j)] *np.exp(-decay*k[(i,j)]/100)  #/(k[(i,j)]**(1/4))

        [caps[(i, j)].append(tNet.G_supergraph[i][j]['capacity']) for (i, j) in tNet.G_supergraph.edges()]
        psi = sum([i ** 2 for i in d.values()])
        obj_ = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        psi_v.append(psi)
        ns.append(len(d))
        print('i: ' + str(cnt) + ' n: ' + str(len(d)) + ', psi: ' + str(psi) + ', obj: ' + str(obj_))
        x.append(cnt)
        cnt += 1
    pool.close()
    return obj, psi_v, ns, x, caps

def get_pwlinear(x,y,breaks):
    # initialize piecewise linear fit with your x and y data
    my_pwlf = pw.PiecewiseLinFit(x, y)

    # fit the data with the specified break points
    # (ie the x locations of where the line segments
    # will terminate)
    my_pwlf.fit_with_breaks(x0)

    # predict for the determined points
    xHat = np.linspace(min(x), max(x), num=20000)
    yHat = my_pwlf.predict(xHat)

    # plot the results
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(xHat, yHat, '-')
    plt.show()

def eval_tt_funct(flow, t0, m, fcoeffs):
    return flow*t0*sum([fcoeffs[n] * (flow/m)**(n) for n in range(len(fcoeffs))])

def get_arc_pwfunc(tNet, i,j, plot=0):
    cap_per_lane = tNet.G_supergraph[i][j]['max_capacity'] / tNet.G_supergraph[i][j]['max_lanes']
    x = np.linspace(cap_per_lane, tNet.G_supergraph[i][j]['max_capacity'], 2000)
    y = [eval_tt_funct(tNet.G_supergraph[i][j]['flow'], tNet.G_supergraph[i][j]['t_0'], m, tNet.fcoeffs) for m in x]
    my_pwlf = pw.PiecewiseLinFit(x, y)
    N = int(tNet.G_supergraph[i][j]['max_lanes']) 
    breaks = [(i+1)*cap_per_lane for i in range(N)]
    x_force = breaks
    y_force = [eval_tt_funct(tNet.G_supergraph[i][j]['flow'], tNet.G_supergraph[i][j]['t_0'], b, tNet.fcoeffs) for b in breaks]

    my_pwlf.fit_with_breaks_force_points(breaks, x_force, y_force)

    if plot == 1 :
        fig, ax = plt.subplots(figsize=(4,2))
        #yHat = eval_pw(x,beta)
        yHat = my_pwlf.predict(x)
        x0 = [i/1500 for i in x]
        p = ax.plot(x0, y, '-', label='$J_{ij}(z_{ij})$')
        p = ax.plot(x0, yHat, '-', label='$\hat{J}_{ij}(z_{ij})$')
        color = p[0].get_color()
        j=0
        for th in breaks:
            if th > min(x) and th< max(x):
                j+=1
                ax.text(th/1500+0.08, yHat[0]-7 ,'$\\theta_{ij}^{('+str(j)+')}$' ,color=color)
                ax.axvline(x=th/1500, linestyle=':', color=color, linewidth=0.75)
            ax.scatter([th/1500], [my_pwlf.predict(th)[0]], marker='.', color=color, s=18)#, s=10)
        ax.set_xlabel('$z_{ij}$')
        ax.set_ylabel('$J_{ij}$')
        ax.set_xlim(min(x0),max(x0))
        plt.legend(frameon=True, framealpha=1,  loc='upper right')
        plt.tight_layout()
        plt.savefig('approx.pdf')
        plt.show()
        plt.cla()
    return my_pwlf.beta, my_pwlf.slopes, breaks

def iteration_ILP(tNet):

    for i in range(5):
        betas = {}
        for i, j in tNet.G_supergraph.edges():
            beta0, beta1, breaks = get_arc_pwfunc(tNet, i, j)
            betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}

        obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        print(obj)
        sol = solve_opt_int_pwl(tNet, betas=betas, breaks=breaks)
        for i, j in tNet.G.edges():
            tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500
        # print new obj
        tNet.solveMSA_supergraph()

def project_fluid_sol_to_integer(tNet):
    v = []
    for i,j in tNet.G_supergraph.edges():
        lanes = round(tNet.G_supergraph[i][j]['capacity']/1500)
        if lanes == 0:
            lanes = 1
            v.append((j,i))
        tNet.G_supergraph[i][j]['lanes'] = lanes
    for i,j in v:
        tNet.G_supergraph[i][j]['lanes'] -= 1
    for i,j in tNet.G_supergraph.edges():
        tNet.G_supergraph[i][j]['capacity'] = tNet.G_supergraph[i][j]['lanes']*1500
    return tNet

def solve_optimal_ILP(tNet):
    # solve optimal ILP
    betas = {}
    breaks = {}
    for i, j in tNet.G_supergraph.edges():
        beta0, beta1, breaks0 = get_arc_pwfunc(tNet, i, j)
        betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}
        breaks[(i, j)] = breaks0
    sol = solve_opt_int_pwl(tNet, betas=betas, breaks=breaks)
    for i, j in tNet.G.edges():
        tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500

def comparison_ILP_fluid(tNet):
    tNeta = copy.deepcopy(tNet)
    tNetc = copy.deepcopy(tNet)
    # solve original
    obj0 = get_obj(tNetc.G_supergraph, tNetc.fcoeffs)
    # solve fluidic
    objf, psi_v, ns, x, caps, tNetc = solve_opt_fluid(tNetc, sequential=False, psi=9999, eps=1e-10)
    objfluid = get_obj(tNetc.G_supergraph, tNetc.fcoeffs)
    # easy map fluidic to integer
    project_fluid_sol_to_integer(tNetc)
    objfluidInt = get_obj(tNetc.G_supergraph, tNetc.fcoeffs)
    # solve optimal ILP problem
    solve_optimal_ILP(tNeta)
    objILP = get_obj(tNeta.G_supergraph, tNeta.fcoeffs)

    return obj0, objILP, objfluid, objfluidInt

def compare_diff_g(tNet0, gs, n_lines_CARS):
    obj0v = []
    objILPv = []
    objfluidv = []
    objfluidIntv = []
    for g in gs:
        tNet = copy.deepcopy(tNet0)
        g_per = tnet.perturbDemandConstant(tNet.g, g)
        tNet.set_g(g_per)
        #tNet.build_supergraph(identical_G=True)
        #tNet.solveMSA_supergraph()
        tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False, rebalancing=False,linear=False, bush=True)
        obj0, objILP, objfluid, objfluidInt = comparison_ILP_fluid(tNet)
        print([obj0, objILP, objfluid, objfluidInt])
        obj0v.append((obj0/objILP-1)*100)
        objILPv.append((objILP/objILP-1)*100)
        objfluidv.append((objfluid/objILP-1)*100)
        objfluidIntv.append((objfluidInt/objILP-1)*100)
    return  obj0v, objILPv, objfluidv, objfluidIntv

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

def solve_opt_int_pwl(tNet, betas, breaks, max_lanes='No'):
    # Start model
    m = Model('LA')
    m.setParam('OutputFlag',0)
    #m.setParam('BarHomogeneous', 0)

    # Define variables
    [[m.addVar(lb = 0, name='e^'+str(n)+'_'+str(i)+'_'+str(j)) for n in range(len(betas[(i, j)]['beta1']))] for i, j in tNet.G_supergraph.edges()]
    m.update()
    l = {(i, j): quicksum(m.getVarByName('e^' + str(n) + '_' + str(i) + '_' + str(j)) for n in range(len(betas[(i, j)]['beta1']))) for i, j in tNet.G_supergraph.edges()}
    m.update()

    # Define objective
    obj = 0
    for i,j in tNet.G_supergraph.edges():
        obj += betas[(i, j)]['beta0'][0]
        for n in range(len(betas[(i, j)]['beta1'])):
            obj += betas[(i, j)]['beta1'][n] * m.getVarByName('e^'+str(n)+'_'+str(i)+'_'+str(j))

    # Define constraints
    #print([tNet.G_supergraph[i][j]['max_capacity'] for i, j in tNet.G_supergraph.edges()])
    [m.addConstr(l[(i,j)] >= 1500) for i, j in tNet.G_supergraph.edges()]
    [m.addConstr(l[(i,j)]+l[(j,i)] <= tNet.G_supergraph[i][j]['max_capacity']) for i, j in tNet.G_supergraph.edges()]


    #print(breaks)
    #print(betas[])
    for i, j in tNet.G_supergraph.edges():
        for n in range(len(betas[(i, j)]['beta1'])):
            m.addConstr(quicksum(m.getVarByName('e^' + str(k) + '_' + str(i) + '_' + str(j)) for k in range(n) ) <= breaks[(i,j)][n])
            #a = 0
            #a = quicksum(m.getVarByName('e^' + str(n + k + 1) + '_' + str(i) + '_' + str(j)) for k in range(len(betas[(i, j)]['beta1'])-n-1))
            #m.addConstr(m.getVarByName('e^' + str(n) + '_' + str(i) + '_' + str(j)) + a >= l[(i,j)] -breaks[n])

    m.update()

    # Define number of lane reversals constraint. 
    if max_lanes!='No':
        # add new set of slack variables
        [m.addVar(lb = 0, name='s'+'_'+str(i)+'_'+str(j)) for i, j in tNet.G_supergraph.edges()]
        m.update()
        [m.addConstr(m.getVarByName('s'+'_'+str(i)+'_'+str(j)) >= (l[(i,j)]/1500 - int(tNet.G_supergraph[i][j]['lanes'])))  for i, j in tNet.G_supergraph.edges()]
        [m.addConstr(m.getVarByName('s'+'_'+str(i)+'_'+str(j)) >= -(l[(i,j)]/1500 - int(tNet.G_supergraph[i][j]['lanes']))) for i, j in tNet.G_supergraph.edges()]
        m.addConstr(quicksum(m.getVarByName('s'+'_'+str(i)+'_'+str(j)) for i, j in tNet.G_supergraph.edges()) <= max_lanes*2)
        #print(max_lanes)
        m.update()

    # Use all lanes!!!


    # Solve Problem
    m.setObjective(obj, GRB.MINIMIZE)
    m.update()
    m.optimize()    

    # Get solution
    sol = {(i,j): sum([m.getVarByName('e^' + str(n) + '_' + str(i) + '_' + str(j)).X/1500 for n in range(len(betas[(i, j)]['beta1']))]) for i,j in tNet.G_supergraph.edges()}

    # print original
    #print('original:')
    #print({(i,j): int(tNet.G_supergraph[i][j]['lanes']) for i,j in tNet.G_supergraph.edges()})
    # print solution
    #print('solution:')
    #print(sol)
    # print s
    #print('s:')
    #print({(i,j): m.getVarByName('s' + '_' + str(i) + '_' + str(j)).X for i,j in tNet.G_supergraph.edges()})

    return  sol

def get_obj(G, fcoeffs):
    return sum(G[i][j]['flow'] * G[i][j]['t_0'] * sum([fcoeffs[n] * (G[i][j]['flow'] / G[i][j]['capacity'])**n for n in range(len(fcoeffs))]) for i,j in G.edges())

def more_and_more_lanes(tNetc, max_lanes_vec, gmult=1, n_lines_CARS=5):
    tNet = copy.deepcopy(tNetc)
    objs = []
    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False, rebalancing=False,linear=False, bush=True)
    obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
    for m in max_lanes_vec:
        betas = {}
        breaks = {}
        for i, j in tNet.G_supergraph.edges():
            beta0, beta1, breaks0 = get_arc_pwfunc(tNet, i, j)
            betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}
            breaks[(i, j)] = breaks0
        sol = solve_opt_int_pwl(tNet, betas=betas, breaks=breaks, max_lanes=m)
        for i, j in tNet.G.edges():
            tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500
            tNetc.G_supergraph[i][j]['t_k'] = cars.travel_time(tNet, i, j)
        obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        print('max lane changes: ' + str(m)+', obj: ' + str(obj))
        objs.append(obj)
    return objs

def integralize_inputs(tNet):
    for i, j in tNet.G_supergraph.edges():
        try:
            tNet.G_supergraph[j][i]
        except:
            tNet.G_supergraph.add_edge(j, i, capacity=0, t_0=tNet.G_supergraph[i][j]['t_0'], length=tNet.G_supergraph[i][j]['length'])
    nx.set_edge_attributes(tNet.G_supergraph, 0, 'lanes')
    nx.set_edge_attributes(tNet.G_supergraph, 0, 'max_lanes')
    nx.set_edge_attributes(tNet.G_supergraph, 0, 'max_capacity')
    for i, j in tNet.G_supergraph.edges():
        tNet.G_supergraph[i][j]['lanes'] = max(round(tNet.G_supergraph[i][j]['capacity'] / 1500), 1)
        if tNet.G_supergraph[j][i] == False:
            print('a')
    for i, j in tNet.G_supergraph.edges():
        tNet.G_supergraph[i][j]['capacity'] = tNet.G_supergraph[i][j]['lanes'] * 1500
        tNet.G_supergraph[i][j]['max_capacity'] = (tNet.G_supergraph[i][j]['lanes'] + tNet.G_supergraph[j][i]['lanes']) * 1500
        tNet.G_supergraph[i][j]['max_lanes'] = tNet.G_supergraph[i][j]['lanes'] + tNet.G_supergraph[j][i]['lanes']
    maxcaps = {(i, j): tNet.G_supergraph[i][j]['capacity'] + tNet.G_supergraph[j][i]['capacity'] for i, j in tNet.G_supergraph.edges()}
    return tNet, maxcaps

def solve_alternating(tNet0, g_per, e=1e-2, type_='full', n_lines_CARS=5):
    tNet = copy.deepcopy(tNet0)
    tNet.set_g(g_per)
    eps = 10000
    k = 0
    objs =[]
    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False,
                                                    rebalancing=False, linear=False, bush=True)
    obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
    objs.append(obj)
    while eps>e and k<2:
        betas = {}
        breaks = {}
        for i, j in tNet.G_supergraph.edges():
            beta0, beta1, breaks0 = get_arc_pwfunc(tNet, i, j)
            betas[(i, j)] = {'beta0': beta0, 'beta1': beta1}
            breaks[(i, j)] = breaks0
        if type_== 'one by one':
            sol = solve_opt_int_pwl(tNet, betas=betas, breaks=breaks, max_lanes=1)
        elif type_== 'full':
            sol = solve_opt_int_pwl(tNet, betas=betas, breaks=breaks, max_lanes=len(tNet.G_supergraph.edges()))
        else:
            sol = solve_opt_int_pwl(tNet, betas=betas, breaks=breaks, max_lanes=type_)
        for i, j in tNet.G.edges():
            tNet.G_supergraph[i][j]['capacity'] = sol[(i, j)] * 1500
            tNet.G_supergraph[i][j]['t_k'] = cars.travel_time(tNet, i, j)
        obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        k+=1
        tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=8, exogenous_G=False,
                                                        rebalancing=False, linear=False, bush=True)
        objs.append(obj)
    return tNet, objs

def solve_FW(tNet_, step, ax, n_iter):
    tNet1 = copy.deepcopy(tNet_)
    #TT, d_norm, runtime = tNet1.solveMSA(exogenous_G=False)
    TT, d_norm, runtime = tNet1.solveMSAsocial_capacity_supergraph(build_t0=False, exogenous_G=False, d_step=step, n_iter = n_iter)
    #TT, d_norm, runtime = tNet1.solveMSAsocial_supergraph(build_t0=False, exogenous_G=False)
    ax[0].plot(list(range(len(TT))), TT, label='step='+str(step))
    ax[1].plot(list(range(len(TT))), d_norm, label='step=' + str(step))

    tNet1 = project_fluid_sol_to_integer(tNet1)

    tNet1, runtime, od_flows = cars.solve_bush_CARSn(tNet1, fcoeffs=tNet1.fcoeffs, n=9, exogenous_G=False,
                                                    rebalancing=False, linear=False, bush=True)
    obj = get_obj(tNet1.G_supergraph, tNet1.fcoeffs)
    return tNet1, obj, TT, d_norm


net_name = 'EMA_mid'
#net_name = 'Anaheim'
#net_name = 'Sioux Falls'
#net_name = 'NYC'
g_mult = 1

# Read network
tNet, fcoeffs = read_net(net_name)
dir_out = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + 'contraflow'

tNet.build_supergraph(identical_G=True)
tNet.read_node_coordinates('data/pos/' + net_name + '.txt')
#fig, ax = plt.subplots()
#tnet.plot_network(tNet.G, ax)# width=0.3)
#plt.show()

# Multiply demand
g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
tNet.set_g(g_per)

# Preprocessing
tNet, max_caps = integralize_inputs(tNet)

out_dir = 'results/'+dir_out + '_'+ net_name

exps = [0,0,0,0,1]
n_lines_CARS = 5


# FIRST EXPERIMENT (COMPARISON FOR DIFFERENT DEMAND LEVELS)
if exps[0] == 1:
    tNet0 = copy.deepcopy(tNet)
    g_mult = 1
    g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
    tNet0.set_g(g_per)
    #tNet0, runtime, od_flows = cars.solve_bush_CARSn(tNet0, fcoeffs=tNet.fcoeffs, n=4, exogenous_G=False,
    #                                                 rebalancing=False, linear=False, bush=True)
    gs = [0.5,1,1.5,2,2.5,3]
    gs = [0.25, 0.5, 0.75]
    gs = [1]
    objs = compare_diff_g(tNet0, gs, n_lines_CARS)
    obj0, objILP, objfluid, objfluidInt = objs
    fig, ax  = plot_diff_gs(gs, obj0, objILP, objfluid, objfluidInt)
    mkdir_n(out_dir)
    plt.savefig(out_dir+'/relative_gap.pdf')
    zdump(objs, out_dir + '/objectives.pkl')

# SECOND EXPERIMENT (RESTRICT NUMBER OF LANES)
if exps[1] == 1:
    tNet0 = copy.deepcopy(tNet)
    g_mult = 0.25
    g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
    tNet0.set_g(g_per)
    max_lanes_vec = [i for i in range(40)]
    objs = more_and_more_lanes(tNet0, max_lanes_vec, gmult=g_mult, n_lines_CARS=n_lines_CARS)
    mkdir_n(out_dir)
    zdump(objs, out_dir + '/objectives.pkl')
    zdump(max_lanes_vec, out_dir + '/max_lanes_vec.pkl')
    fig, ax = plt.subplots()
    plt.plot(max_lanes_vec, objs, marker='.')
    plt.xlabel('Maximum number of reversals')
    plt.ylabel('Objective, $\\hat{J}$')
    plt.xlim([0, max(max_lanes_vec)])
    plt.tight_layout()
    plt.savefig(out_dir+'/sparse_LA.pdf')

# THIRD EXPERIMENT (GET OD DEMANDS BENEFITS)
if exps[2] == 1:
    tNet0 = copy.deepcopy(tNet)
    g_mult = 0.25
    g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
    tNet0.set_g(g_per)
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
        tt_imp[od] = (tt_org[od]-tt_new[od])/tt_org[od]*100
        demandVec.append(tNet0.g[od])
        ttVec.append(tt_imp[od])
    for a in tNet0.G_supergraph.edges():
        tt_news[a] = tNet0.G_supergraph[a[0]][a[1]]['t_k']
        tt_impr[a] = (tt_orgs[a]-tt_news[a])/tt_orgs[a]*100
        flowVec.append(tNet0.G_supergraph[a[0]][a[1]]['flow'])
        #if tNet.G_supergraph[a[0]][a[1]]['flow'] > 8000 and tNet.G_supergraph[a[0]][a[1]]['flow'] < 9000:
        if tt_impr[a]>40:
            colorsVec.append('lime')
            a0 = a
        else:
            colorsVec.append('b')
        ttVecs.append(tt_impr[a])

    idx = 0
    for a in tNet.G_supergraph.edges():
        if a ==(a0[1],a0[0]):
            colorsVec[idx] = 'lime'
        idx +=1


    def plot_LR(x,y, c=False):
        x = np.array(x)
        y = np.array(y)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x + b, color='red', linestyle='--')
        plt.scatter(x, y, s=18, alpha=1, c=c)


    mkdir_n(out_dir)

    # PLOT RESULTS PER LINK
    fig, ax = plt.subplots(figsize=(4,2))
    plt.axhline(0, linestyle=':', color='k')
    plot_LR(x=flowVec, y=ttVecs, c=colorsVec)
    plt.xlim(0, max(flowVec))
    #plt.ylim(-5, 5)
    plt.xlabel('Flow')
    plt.ylabel('Improvement (\\%)')
    plt.tight_layout()
    plt.savefig(out_dir + '/link_scatter.pdf')
    #plt.show()

    plt.cla()
    fig, ax = plt.subplots(figsize=(4,2))
    plt.hist(ttVecs, bins=20)
    plt.xlabel('Improvement (\\%)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(out_dir + '/link_hist.pdf')
    #plt.show()

    '''
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
    '''
    zdump(tt_org, out_dir + '/travel_time_original.pkl')
    zdump(tt_new, out_dir + '/travel_time_new.pkl')
    zdump(tt_imp, out_dir + '/travel_time_improvements.pkl')
    #print(tt_org)
    #print(tt_new)
    #print(tt_imp)

# FOURTH EXPERIMENT (ALGORITHM COMPARISON)
if exps[3] == 1:
    objs = {}
    tNet0 = copy.deepcopy(tNet)
    mkdir_n(out_dir)
    for g_mult in [1, 1.5, 2.0, 2.5]:
        g_per = tnet.perturbDemandConstant(tNet0.g, g_mult)
        tNet.set_g(g_per)
        #print(min([tNet.G_supergraph[i][j]['max_capacity'] for i,j in tNet.G_supergraph.edges()]))
        objs[g_mult] = []
        objs_labels = ['Nominal']
        tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False,
                                                        rebalancing=False, linear=False, bush=True)
        #print(min([tNet.G_supergraph[i][j]['max_capacity'] for i, j in tNet.G_supergraph.edges()]))
        obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        objs[g_mult].append(obj)
        print('Nominal:' + str(obj))

        fig, ax = plt.subplots(2, sharex=True)
        steps = [5e0, 1e1, 5e1, 1e2, 5e2, 1e3]
        n_iter = 1000
        for step in steps:
            tNet_ = copy.deepcopy(tNet0)
            tNet_.set_g(g_per)
            tNet_, obj, TT, dnorm = solve_FW(tNet_, step, ax, n_iter=n_iter)
            print('FW ('+str(step) +') : ' + str(obj))
            objs_labels.append('FW: '+str(step))
            objs[g_mult].append(obj)
            zdump(TT, out_dir + '/FW_step'+str(step)+'_mult_' + str(g_mult) + '.pkl')
            zdump(dnorm, out_dir + '/FW_step' + str(step) + '_mult_' + str(g_mult) + '.pkl')
        ax[0].set_xlim((0, n_iter))
        plt.legend()
        plt.tight_layout()
        ax[1].set_xlim((0, n_iter))
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Objective')
        ax[0].set_yscale('log')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Derivative norm')
        ax[1].set_yscale('log')
        plt.savefig(out_dir + '/FW_iteration_mult'+str(g_mult)+'.pdf')

        fig, ax = plt.subplots()
        _, obj1b1 = solve_alternating(tNet0, g_per=g_per, e=1e-2, type_='one by one', n_lines_CARS=n_lines_CARS)
        print('one: ' + str(obj1b1[-1]))
        _, obj5 = solve_alternating(tNet0, g_per=g_per, e=1e-2, type_= 5, n_lines_CARS=n_lines_CARS)
        print('five: ' + str(obj5[-1]))
        _, objfull = solve_alternating(tNet0, g_per=g_per, e=1e-2, type_='full', n_lines_CARS=n_lines_CARS)
        print('all: ' + str(objfull[-1]))

        objs_labels.append('One')
        objs_labels.append('Five')
        objs_labels.append('All')
        objs[g_mult].append(obj1b1[-1])
        objs[g_mult].append(obj5[-1])
        objs[g_mult].append(objfull[-1])

        zdump(obj1b1, out_dir + '/iteration_one_mult_'+str(g_mult)+'.pkl')
        zdump(obj5, out_dir + '/iteration_five_mult_'+str(g_mult)+'.pkl')
        zdump(objfull, out_dir + '/iteration_all_mult_'+str(g_mult)+'.pkl')

        ax.plot(obj1b1, label='One', marker='o')
        ax.plot(obj5, label='Five', marker='o')
        ax.plot(objfull, label='All', marker='o')
        ax.set_xlim((0,len(obj5)))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')

        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir + '/sequential_iteration_mult'+str(g_mult)+'.pdf')
        print(objs)

    zdump(objs, out_dir + '/obj_results.pkl')

# FIFTH EXPERIMENT (PLOTTING)
if exps[4] == 1:
    tNet0 = copy.deepcopy(tNet)
    g_mult = 3.5
    g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
    tNet.set_g(g_per)
    nominal_capacity = {(i,j):tNet0.G_supergraph[i][j]['capacity'] for i,j in tNet0.G_supergraph.edges()}

    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False,
                                                    rebalancing=False, linear=False, bush=True)

    edges, weights = zip(*nx.get_edge_attributes(tNet.G_supergraph, 'flow').items())

    weights = [tNet.G_supergraph[i][j]['flow'] * tNet.G_supergraph[i][j]['t_k'] for i, j in edges]
    min_w  = min(weights)
    max_w = max(weights)

    tNet0, objfull = solve_alternating(tNet0, g_per=g_per, e=1e-2, type_='full', n_lines_CARS=n_lines_CARS)

    opt_capacity = {(i, j): tNet0.G_supergraph[i][j]['capacity'] for i,j in tNet0.G_supergraph.edges()}
    change_edges = [(i, j) for i, j in tNet0.G_supergraph.edges() if nominal_capacity[(i, j)] < opt_capacity[(i, j)]]
    weights2 = [tNet.G_supergraph[i][j]['flow']*tNet.G_supergraph[i][j]['t_k'] for i, j in edges if nominal_capacity[(i, j)] < opt_capacity[(i, j)]]

    node_demand = [sum(v for k,v in tNet0.g.items() if k[1] == n) for n in tNet0.G.nodes()]
    node_size = [x/max(node_demand)*2 for x in node_demand]


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,2.5))
    cmap = plt.cm.plasma_r
    normalize = matplotlib.colors.Normalize(vmin=min_w, vmax=max_w)
    tnet.plot_network(tNet0.G, ax[0], edgecolors=weights, edge_width=0.3, cmap=cmap, nodesize=0.1, vmin=min_w, vmax=max_w)
    tnet.plot_network(tNet0.G, ax[0], edgelist=change_edges, edgecolors=weights2, edge_width=0.4, arrowsize=4, cmap=cmap, nodesize=0.1, vmin=min_w, vmax=max_w)

    weights3 = [tNet0.G_supergraph[i][j]['flow'] * tNet0.G_supergraph[i][j]['t_k'] for i, j in edges]
    tnet.plot_network(tNet0.G, ax[1], edgecolors=weights3, edge_width=0.3, cmap=cmap, nodesize=0.1,vmin=min_w, vmax=max_w)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    fig.colorbar(sm, ax=ax.ravel().tolist())
    plt.savefig('hola.pdf')

