import time
import networkx as nx
import src.tnet as tnet
import copy
import src.CARS as cars
from multiprocessing import Pool
import multiprocessing as mp
from src.utils import *
import  pwlf as pw
from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

plt.style.use(['science','ieee', 'high-vis'])

def eval_tt_funct(flow, t0, m, fcoeffs):
    return flow*t0*sum([fcoeffs[n] * (flow/m)**(n) for n in range(len(fcoeffs))])

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
def move_capacity(G, d, i, gamma, gamma0):
    V = []
    l = i
    for (i, j), g in d.items():
        v = gamma[(i, j)]/l * g
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

def bisection_capacity(G, m_derivative, bisection_n=10, method_='bisection'):
    if method_ == 'bisection':
        l = 1 / 2
        ub = 1
        lb = 0
        for n in range(bisection_n):  # Solve issue with exogenous
            f = 0
            for edge in G.edges():
                m = nx.get_edge_attribute(edge, 'capacity')
                if m_derivative[edge[0], edge[1]] > 0:
                    m_star = m + 1500
                else:
                    m_star = m - 1500
                m_target = l*m_star - (1-l)*m
                f += get_travel_time_social(
                    x=self.get_edge_attribute(edge, 'flow'),
                    m=m_target,
                    t0=self.get_edge_attribute(edge, 't_0'),
                    fcoeffs=self.fcoeffs)\
                    * (self.get_edge_attribute(edge, 'help_flow') - self.get_edge_attribute(edge, 'flow'))
            #print(f)
            if f > 0:
                ub = l
            else:
                lb = l
            l = (ub - lb) / 2
        #print(l)
        return l

def solve_opt_fluid(tNet, sequential=False, psi = 9999, eps = 1e-10):
    gamma0 = {(i, j): 1 for i, j in tNet.G_supergraph.edges()}
    gamma = copy.deepcopy(gamma0)
    #k = {(i, j): 1 for i, j in tNet.G.edges()}
    k=0
    caps = {(i, j): [] for i, j in tNet.G_supergraph.edges()}

    obj = []
    psi_v = []
    ns = []
    x = []
    cnt = 0
    delta = 20
    decay = 0.001
    pool = mp.Pool(mp.cpu_count() - 1)
    i = 0
    while psi >= eps and cnt < 5000:
        i += 1
        if sequential:
            #tNet.solveMSA()
            tNet.tNet.solveMSAsocial_supergraph()
            #tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=8, exogenous_G=False, rebalancing=False, linear=False, bush=True)
        d, b = get_derivative(tNet.G_supergraph, tNet.fcoeffs, delta, pool)
        tNet.G_supergraph, v = move_capacity(tNet.G_supergraph, d, i, gamma, gamma0)


        psi = sum([i ** 2 for i in d.values()])
        obj_ = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
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

def solve_greedy(tNet, betas, breaks, max_lanes=None, max_links=None):
    G = tNet.G_supergraph
    avail_lanes = sum([G[i][j]['lanes'] for i, j in G.edges()])
    obj = 0
    active_beta = {}
    for i, j in G.edges():
        G[i][j]['lanes'] = 1
        avail_lanes -= 1
        obj += betas[(i,j)]['beta1'][0]
        active_beta[(i,j)] = 1

    avail_roads = []
    for i, j in G.edges():
        l = G[i][j]['lanes'] + G[j][i]['lanes']
        if l < G[i][j]['max_lanes']:
            avail_roads.append((i,j))

    while avail_lanes > 0:
        d = {(i, j): betas[(i, j)]['beta1'][active_beta[(i, j)]] for i, j in avail_roads}
        k = min(d, key=d.get)
        i, j = k
        obj += betas[k]['beta1'][active_beta[k]]
        active_beta[k] += 1
        avail_lanes -= 1

        G[i][j]['lanes'] += 1
        l = G[i][j]['lanes'] + G[j][i]['lanes']
        if l == G[i][j]['max_lanes']:
            avail_roads.remove((i, j))
            avail_roads.remove((j, i))
        print(avail_roads)
        #print(k)
        #pause()
        #max([betas[(i,j)]['beta1'][active_beta[(i,j)]] for i,j in G.edges()])
        #d = {(i,j): G[i][j]['lanes'] for i, j in G.edges()}

        #active_beta[(ii, jj)] += 1
        #max()


def solve_opt_int_pwl(tNet, betas, breaks, max_lanes=None, max_links=None):
    # Start model
    m = Model('LA')
    m.setParam('OutputFlag', 0)
    #m.setParam('BarHomogeneous', 0)
    # Define variables
    G = tNet.G_supergraph
    edges = G.edges()
    eps = {}
    for i, j in edges:
        for n in range(len(betas[(i, j)]['beta1'])):
            #eps[(i, j, n)] = m.addVar(lb=0, ub=1, name='e^' + str(n) + '_' + str(i) + '_' + str(j))
            eps[(i, j, n)] = m.addVar(vtype=GRB.BINARY, name='e^' + str(n) + '_' + str(i) + '_' + str(j))
    m.update()

    l = {(i, j): quicksum(eps[(i, j, n)] for n in range(len(betas[(i, j)]['beta1']))) for i, j in edges}
    m.update()

    # Define objective
    obj = 0
    for i, j in edges:
        #obj += betas[(i, j)]['beta0'][0]
        for n in range(len(betas[(i, j)]['beta1'])):
            obj += betas[(i, j)]['beta1'][n] * eps[(i, j, n)]

    # Define constraints
    [m.addConstr(l[(i, j)] >= 1) for i, j in edges]
    [m.addConstr(l[(i, j)] + l[(j, i)] == G[i][j]['max_lanes']) for i, j in edges]

    for i, j in edges:
        for n in range(len(betas[(i, j)]['beta1'])):
            m.addConstr(quicksum(eps[(i, j, k)] for k in range(n)) <= breaks[(i,j)][n])
            if n > 0:
                m.addConstr(eps[(i, j, n)] <= eps[(i, j, n-1)])
            #a = quicksum(eps[(i, j, n+k+1)] for k in range(len(betas[(i, j)]['beta1'])-n-1))
            #m.addConstr(eps[(i, j, n)] + a >= l[(i,j)] - breaks[n])

    m.update()

    # Define number of lane reversals constraint.
    if max_lanes != None:
        # add new set of slack variables
        s = {(i, j): m.addVar(lb=0, name='s'+'_'+str(i)+'_'+str(j)) for i, j in edges}
        m.update()
        [m.addConstr(s[(i, j)] >= (l[(i,j)] - G[i][j]['lanes'])) for i, j in edges]
        [m.addConstr(s[(i, j)] >= -(l[(i,j)] - G[i][j]['lanes'])) for i, j in edges]
        m.addConstr(quicksum(s[(i, j)] for i, j in edges) <= max_lanes*2)
        #print(max_lanes)

    if max_links != None:
        # add new set of slack variables
        m.addVar(lb=0, name='zeta')
        m.update()
        Cnstr = 0

        #a = {(i, j): m.addVar(lb=0, name='a' + str(i) + '_' + str(j)) for i, j in edges}
        for i, j in tNet.G_supergraph.edges():
            m.update()
            n0 = tNet.G_supergraph[i][j]['lanes']
            Cnstr += 1 - eps[(i, j, n0-1)]
            #m.addConstr(a[(i, j)] >= 1 - eps[(i, j, n0-1)])
        #m.addConstr(m.getVarByName('zeta') >= quicksum(a[(i, j)] for i, j in edges))
        m.addConstr(m.getVarByName('zeta') >= Cnstr)
        m.addConstr(m.getVarByName('zeta') <= max_links)

        #obj += m.getVarByName('zeta')*Lambda

    # Solve Problem
    m.setObjective(obj, GRB.MINIMIZE)
    m.update()
    m.optimize()

    #print(m.status)
    #print(GRB.OPTIMAL)

    # Get solution
    sol = {(i,j): sum([eps[(i, j, n)].X for n in range(len(betas[(i, j)]['beta1']))]) for i, j in tNet.G_supergraph.edges()}
    arr = [eps[(i, j, n)].X for i,j in tNet.G_supergraph.edges() for n in range(len(betas[(i, j)]['beta1']))]
    arr = np.array(arr)
    #print(np.unique(arr))
    #pause
    # print original
    #print('original:')
    #print({(i,j): int(tNet.G_supergraph[i][j]['lanes']) for i,j in tNet.G_supergraph.edges()})
    # print solution
    #print('solution:')
    #print(sol)
    # print s
    #print('s:')
    #print({(i,j): m.getVarByName('s' + '_' + str(i) + '_' + str(j)).X for i,j in tNet.G_supergraph.edges()})

    return sol


def solve_FW(tNet_, step, n_iter):
    tNet1 = copy.deepcopy(tNet_)
    #TT, d_norm, runtime = tNet1.solveMSA(exogenous_G=False)
    TT, d_norm, runtime, RG = tNet1.solveMSAsocial_capacity_supergraph(build_t0=False,
                                                                       exogenous_G=False,
                                                                       d_step=step,
                                                                       n_iter=n_iter)
    #TT, d_norm, runtime = tNet1.solveMSAsocial_supergraph(build_t0=False, exogenous_G=False)
    #ax[0].plot(list(range(len(TT))), TT, label='step='+str(step), alpha=0.7)
    #ax[1].plot(list(range(len(TT))), d_norm, label='step=' + str(step), alpha=0.7)
    #ax[2].plot(list(range(len(RG))), RG, label='step=' + str(step), alpha=0.7)

    tNet1 = project_fluid_sol_to_integer(tNet1)

    #tNet1, runtime, od_flows, _ = cars.solve_bush_CARSn(tNet1,
    #                                                 fcoeffs=tNet1.fcoeffs,
    #                                                 n=9,
    #                                                 exogenous_G=False,
    #                                                 rebalancing=False,
    #                                                 linear=False,
    #                                                 bush=True)
    obj = tnet.get_totalTravelTime(tNet1.G_supergraph, tNet1.fcoeffs)
    c = {(i,j): tNet1.G_supergraph[i][j]['capacity'] for i,j in tNet1.G_supergraph.edges()}
    return tNet1, obj, TT, d_norm, RG, c, runtime

def get_obj_pwl(tnet, e, xu, s, theta, a, lambda_cap, QP=False):
    obj  = 0
    for i,j in tnet.G_supergraph.edges():
        t0 = tnet.G_supergraph[i][j]['t_0']
        m = tnet.G_supergraph[i][j]['capacity']
        obj += t0 * xu[(i, j)]/m
        for l in range(len(theta)-1):
            obj +=  t0*(a[l]/m) * e[(l, i, j)]
    if QP == False:
        obj += lambda_cap * quicksum(s[(i, j)] for i, j in tnet.G.edges())
    else:
        obj += lambda_cap * quicksum(s[(i, j)] * s[(i, j)] for i, j in tnet.G.edges())
    return obj


def solve_alternating(tNet0, e=1e-2, theta=None, a=None, type_='full', n_lines_CARS=5):
    tNet = copy.deepcopy(tNet0)
    #tNet.set_g(g_per)
    eps = 10000
    k = 0
    objs =[]

    t0 = time.time()
    tNet, runtime, od_flows, c = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False,
                                                    rebalancing=False, linear=False, bush=True, theta=theta, a=a)
    obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
    objs.append(obj)
    while eps>e and k<7:
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
        obj = tnet.get_totalTravelTime(tNet.G_supergraph, tNet.fcoeffs)
        k+=1
        tNet, runtime, od_flows, c = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=n_lines_CARS, exogenous_G=False,
                                                        rebalancing=False, linear=False, bush=True, theta=theta, a=a)
        objs.append(obj)
    rtime = time.time() - t0
    return tNet, objs, rtime


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


def get_arc_pwfunc(tNet, i,j, plot=0):
    cap_per_lane = tNet.G_supergraph[i][j]['max_capacity'] / tNet.G_supergraph[i][j]['max_lanes']
    #x = np.linspace(cap_per_lane, tNet.G_supergraph[i][j]['max_capacity'], 2000)
    x = np.linspace(1, tNet.G_supergraph[i][j]['max_lanes'])
    y = [eval_tt_funct(tNet.G_supergraph[i][j]['flow'], tNet.G_supergraph[i][j]['t_0'], m*cap_per_lane, tNet.fcoeffs) for m in x]
    my_pwlf = pw.PiecewiseLinFit(x, y)
    N = int(tNet.G_supergraph[i][j]['max_lanes'])
    breaks = [(i+1) for i in range(N)]
    x_force = breaks
    y_force = [eval_tt_funct(tNet.G_supergraph[i][j]['flow'], tNet.G_supergraph[i][j]['t_0'], b*cap_per_lane, tNet.fcoeffs) for b in breaks]

    my_pwlf.fit_with_breaks_force_points(breaks, x_force, y_force)

    if plot == 1 :
        fig, ax = plt.subplots(figsize=(4,2))
        #yHat = eval_pw(x,beta)
        yHat = my_pwlf.predict(x)
        x0 = [i for i in x]
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
        #plt.show()
        plt.cla()
    return my_pwlf.beta, my_pwlf.slopes, breaks
