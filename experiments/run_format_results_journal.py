import matplotlib.pyplot as plt
import pandas as pd
from src.utils import *
from src.utils import *
from os import listdir
plt.style.use(['science','ieee', 'high-vis'])


def plot_diff_gs(files_dir, out_dir):
    objs = zload(files_dir + '/objs_demands.pkl')
    obj0, objILP, objfluid, objfluidInt = objs
    gs = [0.5, 1, 1.5, 2, 2.5, 3]
    fig, ax = plt.subplots(figsize=(4,2))
    ax.plot(gs, objILP, label='LP')
    ax.plot(gs, objfluid, label='Lower bound', marker='.')
    ax.plot(gs, objfluidInt, label='$\\mathcal{P}($Lower bound)', marker='o')
    ax.plot(gs, obj0, label='Original', marker='+')
    plt.xlabel('Demand multiplier')
    plt.ylabel('Deviation from optimal (\\%)')
    plt.xlim((min(gs), max(gs)))
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_dir + '/relative_gap.pdf')



def plot_alg_comparison(files_dir, out_dir):
    fig, ax = plt.subplots(3, sharex=True, figsize=(4, 6))
    files = listdir(files_dir)
    steps = []
    for file in files:
        if 'step' in file:
            steps.append(file.split('_')[1])
    for step in set(steps):
        TT = zload(files_dir + '/step_' + str(step) + '_TT_' + str(g_mult) + '.pkl')
        dnorm = zload(files_dir + '/step_' + str(step) + '_dnorm_' + str(g_mult) + '.pkl')
        RG = zload(files_dir + '/step_' + str(step) + '_RG_' + str(g_mult) + '.pkl')
        ax[0].plot(list(range(len(TT))), TT, ls='-', label='step='+str(step), alpha=0.7)
        ax[1].plot(list(range(len(TT))), dnorm, ls='-', label='step=' + str(step), alpha=0.7)
        ax[2].plot(list(range(len(RG))), RG, ls='-', label='step=' + str(step), alpha=0.7)
        ax[0].set_xlim((0, len(RG)))
    plt.legend()
    ax[1].set_xlim((0, len(RG)))
    ax[2].set_xlim((0, len(RG)))
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Objective')
    ax[0].set_yscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Derivative norm')
    ax[1].set_yscale('log')
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Relative Gap')
    ax[2].set_yscale('log')
    plt.tight_layout()
    plt.savefig(out_dir + '/FW_iteration_mult' + str(g_mult).replace('.', '_') + '.pdf')

def plot_max_num_reversals(files_dir, out_dir):
    objs = zload(files_dir + '/objs_num_lanes.pkl')
    max_lanes_vec = zload(files_dir + '/max_lanes_vec.pkl')
    objs = [i/objs[0] for i in objs]
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.plot(max_lanes_vec, objs, marker='.')
    plt.xlabel('Max. number of reversals, $k$')
    plt.ylabel('$J^{\\text{LP}}_k/J^{\\text{Original}}$')
    plt.xlim([0, max(max_lanes_vec)])
    plt.tight_layout()
    plt.savefig(out_dir+'/sparse_LA.pdf')

def table_comparisons(nets, gmultis):
    out_dir = 'results/'
    res = {}
    for net in nets:
        for mult in gmultis:
            d = zload(out_dir+net+'_'+str(mult)+'/obj_results.pkl')
            res[(net, mult)]= d[mult]
    df = pd.DataFrame.from_dict(res).T
    print(df)
    df = 1/df[[1,2,3,4,5,6,7,8]].div(df[0], axis=0)
    print(df)


if __name__ == "__main__":
    '''
    # PARAMETERS
    net_name = str(sys.argv[1])
    g_mult = float(sys.argv[2])
    # define directories
    ext = net_name + '_' + str(g_mult)
    files_dir = 'tmp/' + ext
    out_dir = 'results/' + ext
    # create a new directory
    mkdir_n(out_dir)
    # generate results
    plot_alg_comparison(files_dir, out_dir)
    plot_diff_gs(files_dir, out_dir)
    plot_max_num_reversals(files_dir, out_dir)
    '''
    # Generate table with results
    nets = ['test_9','EMA_mid']
    gmultis = [1.0, 1.5, 2.0, 3.0]
    table_comparisons(nets, gmultis)



