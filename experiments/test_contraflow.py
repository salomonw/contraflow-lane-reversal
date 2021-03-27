import src.tnet as tnet
import copy
import src.CARS as cars
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#import src.pwapprox as pw
from gurobipy import *
import pwlf as pw
from src.utils import *


plt.style.use(['science','ieee', 'high-vis'])

def get_derivative(tNet, delta):
    # Find derivatives
    tNetC = copy.deepcopy(tNet)
    g = {}
    # Case 1: Find derivative estimates by assuming fixed routes
    d = {}
    for i,j in tNetC.G_supergraph.edges():
        t0 = eval_tt_funct(tNetC, tNetC.G_supergraph[i][j]['capacity'], i, j)
        tinv0 = eval_tt_funct(tNetC, tNetC.G_supergraph[j][i]['capacity'], j, i)

        tNetC.G_supergraph[i][j]['capacity'] += delta#*tNetC.G_supergraph[i][j]['capacity']
        tNetC.G_supergraph[j][i]['capacity'] -= delta#*tNetC.G_supergraph[i][j]['capacity']
        t = eval_tt_funct(tNetC, tNetC.G_supergraph[i][j]['capacity'], i, j)
        tinv = eval_tt_funct(tNetC, tNetC.G_supergraph[j][i]['capacity'], j, i)

        #v = ((t-t0) - (tinv0-tinv))/(delta*tNetC.G_supergraph[i][j]['capacity'])
        v = ((t+tinv)-(t0+tinv0))/(delta)#*tNetC.G_supergraph[i][j]['capacity'])
        #print(v)
        #if v>0:
        d[(i,j)] = v


        tNetC.G_supergraph[i][j]['capacity'] -= delta#*tNetC.G_supergraph[i][j]['capacity']
        tNetC.G_supergraph[j][i]['capacity'] += delta#*tNetC.G_supergraph[i][j]['capacity']

    #print(d)
    if len(d)>0:
        a = max(d, key=d.get)
        return d, a
    else:
        return {}, (0,0)
    #print(a)

def solve_opt_fluid(tNet, sequential=False, psi = 9999, eps = 1e-10):
    #gamma0 = {(i, j): 0.9999 for i, j in tNet.G.edges()}
    gamma0 = {(i, j): 0.9999 for i, j in tNet.G.edges()}
    gamma = gamma0
    k = {(i, j): 1 for i, j in tNet.G.edges()}
    caps = {(i, j): [] for i, j in tNet.G.edges()}

    obj = []
    psi_v = []
    ns = []
    x = []
    cnt = 0
    delta = 0.001
    decay = 0.001
    while psi >= eps and cnt < 5000:
        if sequential:
            #tNet.solveMSA()
            tNet.solveMSA_supergraph()
            #tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=8, exogenous_G=False, rebalancing=False, linear=False, bush=True)
        d, b = get_derivative(tNet, delta)
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
            gamma[(i, j)] = gamma0[(i, j)]#*np.exp(-decay*k[(i,j)]/10)  #/(k[(i,j)]**(1/4))

        [caps[(i, j)].append(tNet.G_supergraph[i][j]['capacity']) for (i, j) in tNet.G_supergraph.edges()]
        psi = sum([i ** 2 for i in d.values()])
        #obj_ = tNet.eval_obj()
        # obj_ = cars.eval_obj_funct(tNet, G_exogenous=False)
        obj_ = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        obj.append(obj_)
        psi_v.append(psi)
        ns.append(len(d))
        print('i: ' + str(cnt) + ' n: ' + str(len(d)) + ', psi: ' + str(psi) + ', obj: ' + str(obj_))
        x.append(cnt)
        cnt += 1
    print('i: ' + str(cnt) + ' n: ' + str(len(d)) + ', psi: ' + str(psi) + ', obj: ' + str(obj_))
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
    while psi >= eps and cnt < 15000:
        if sequential:
            #tNet.solveMSA()
            tNet.solveMSA_supergraph()
            #tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=8, exogenous_G=False, rebalancing=False, linear=False, bush=True)
        d, b = get_derivative(tNet, delta)
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
        #obj_ = tNet.eval_obj()
        # obj_ = cars.eval_obj_funct(tNet, G_exogenous=False)
        obj_ = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        psi_v.append(psi)
        ns.append(len(d))
        print('i: ' + str(cnt) + ' n: ' + str(len(d)) + ', psi: ' + str(psi) + ', obj: ' + str(obj_))
        x.append(cnt)
        cnt += 1
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

def eval_tt_funct(tNet, m, i,j):
    return tNet.G_supergraph[i][j]['flow']*tNet.G_supergraph[i][j]['t_0']*sum([tNet.fcoeffs[n] * (tNet.G_supergraph[i][j]['flow']/m)**(n)  for n in range(len(tNet.fcoeffs))])

def get_arc_pwfunc(tNet, i,j, plot=1):
    cap_per_lane = tNet.G_supergraph[i][j]['max_capacity'] / tNet.G_supergraph[i][j]['max_lanes']
    x = np.linspace(cap_per_lane, tNet.G_supergraph[i][j]['max_capacity'], 2000)
    y = [eval_tt_funct(tNet, m, i,j) for m in x]
    my_pwlf = pw.PiecewiseLinFit(x, y)
    N = int(tNet.G_supergraph[i][j]['max_lanes']) 
    breaks = [(i+1)*cap_per_lane for i in range(N)]
    x_force = breaks
    y_force = [eval_tt_funct(tNet, b, i,j) for b in breaks]

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
        #tNet.solveMSA()
        tNet.solveMSA_supergraph()
        #tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=8, exogenous_G=False,
        #                                                rebalancing=False, linear=False, bush=True)

def project_fluid_sol_to_integer(tNet):
    v = []
    for i,j in tNet.G_supergraph.edges():
        lanes = round(tNet.G_supergraph[i][j]['capacity']/1500,0)
        if lanes ==0:
            lanes = 1
            v.append((j,i))
        tNet.G_supergraph[i][j]['lanes'] = lanes
    for i,j in v:
        tNet.G_supergraph[i][j]['lanes'] = lanes-1
    for i,j in tNet.G_supergraph.edges():
        tNet.G_supergraph[i][j]['capacity'] = tNet.G_supergraph[i][j]['lanes']*1500


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

def compare_diff_g(tNet0, gs):
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
        tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=5, exogenous_G=False, rebalancing=False,linear=False, bush=True)
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
        print(max_lanes)
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



def more_and_more_lanes(tNetc, max_lanes_vec, gmult=1):
    tNet = copy.deepcopy(tNetc)
    objs = []
    tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=5, exogenous_G=False, rebalancing=False,linear=False, bush=True)
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
        obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
        print('max lane changes: ' + str(m)+', obj: ' + str(obj))
        objs.append(obj)
    return objs


net_name = 'EMA_mid'
g_mult = 1

# Read network
netFile, gFile, fcoeffs, tstamp, dir_out = tnet.get_network_parameters(net_name,experiment_name='relative_gap')
tNet = tnet.tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)

# multiply demand
g_per = tnet.perturbDemandConstant(tNet.g, g_mult)
tNet.set_g(g_per)

tNet.build_supergraph(identical_G=True)

# integralize the inputs
maxcaps = {(i,j) : tNet.G_supergraph[i][j]['capacity']+tNet.G_supergraph[j][i]['capacity'] for i,j in tNet.G_supergraph.edges()}
nx.set_edge_attributes(tNet.G_supergraph, 0, 'lanes')
nx.set_edge_attributes(tNet.G_supergraph, 0, 'max_lanes')
nx.set_edge_attributes(tNet.G_supergraph, 0, 'max_capacity')
for i,j in tNet.G_supergraph.edges():
    tNet.G_supergraph[i][j]['lanes'] = max(np.round(tNet.G_supergraph[i][j]['capacity']/1500), 1)
for i,j in tNet.G.edges():
    tNet.G_supergraph[i][j]['capacity'] = tNet.G_supergraph[i][j]['lanes']*1500
    tNet.G_supergraph[i][j]['max_capacity'] = (tNet.G_supergraph[i][j]['lanes'] + tNet.G_supergraph[j][i]['lanes'])*1500
    tNet.G_supergraph[i][j]['max_lanes'] = tNet.G_supergraph[i][j]['lanes'] + tNet.G_supergraph[j][i]['lanes']

print(len(tNet.G_supergraph.edges()))
print(sum(tNet.G_supergraph[i][j]['lanes'] for i,j in tNet.G_supergraph.edges()))
# Solve traffic assignment for actual net


#tNet.solveMSA()
#tNet.solveMSA_supergraph()
#tNet.solveMSAsocial_supergraph()
#tNet, runtime, od_flows = cars.solve_bush_CARSn(tNet, fcoeffs=tNet.fcoeffs, n=5, exogenous_G=False, rebalancing=False,linear=False, bush=True)
#print('TAP done!')

#obj, psi_v, ns, x, caps = solve_opt_fluid(tNet, sequential=True, psi = 9999, eps = 1e-10)
#iteration_ILP(tNet)


#print(tNet.fcoeffs)
#obj0 = get_obj(tNet.G_supergraph, tNet.fcoeffs)
#print(tNet.totalDemand)
#print(obj0/tNet.totalDemand)
#pause

out_dir = 'results/'+dir_out +'_'+ net_name

'''
gs = [0.5,1,1.5,2,2.5,3]
objs = compare_diff_g(tNet, gs)
obj0, objILP, objfluid, objfluidInt = objs
fig, ax  = plot_diff_gs(gs, obj0, objILP, objfluid, objfluidInt)
mkdir_n(out_dir)
plt.savefig(out_dir+'/relative_gap.pdf')
zdump(objs, out_dir + '/objectives.pkl')

#'''
max_lanes_vec = [i for i in range(40)]
objs = more_and_more_lanes(tNet, max_lanes_vec, gmult=g_mult)
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
print(asdf)
#'''
#'''
#comparison_ILP_fluid(tNet)

'''
# print objective
obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
print(obj)

sol = solve_opt_int_pwl(tNet, betas=betas, breaks=breaks)
print(sol)
# assign new capacities
for i,j in tNet.G.edges():
    tNet.G_supergraph[i][j]['capacity'] = sol[(i,j)]*1500

#print new obj
obj = get_obj(tNet.G_supergraph, tNet.fcoeffs)
print(obj)
'''


'''
fig, ax = plt.subplots(3, sharex=True)
ax[0].plot(x, obj)
ax[1].plot(x, psi_v)
ax[2].plot(x, ns)
ax[0].set_ylabel('$F(\\mathbf{x}^*, \\mathbf{m})$')
ax[1].set_ylabel('$||\\Psi||$')
ax[2].set_ylabel('$|\\Psi|$')
plt.xlim((0,len(x)))
plt.xlabel('Iteration')
plt.tight_layout()

fig1, ax1 = plt.subplots(1)
for (i,j) in tNet.G_supergraph.edges():
    ax1.plot(caps[(i,j)], label='('+str(i)+', '+str(j)+')')
ax1.set_ylabel('$\\mathbf{m}$')
plt.xlabel('Iteration')
#plt.legend()
plt.xlim((0,len(x)))
plt.tight_layout()

plt.show()
'''






'''
# Case 2: total  derivatives
obj = tNet.eval_obj()
tNetC = copy.deepcopy(tNet)
g = {}
for a in tNetC.Gd.edges():
    tNetC.G.edge[a] = tNetC.Gd.edge[a] + delta
    tNet.solveMSA()
    g[a] = (obj - tNetC.eval_obj)/delta
    tNetC.Gd.edge[a] = tNetC.Gd.edge[a] - delta
'''