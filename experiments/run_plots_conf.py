import matplotlib.pyplot as plt
from src.utils import *
plt.style.use(['science','ieee', 'high-vis'])


def plot_diff_gs(gs, obj0, objILP, objfluid, objfluidInt):
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
    return fig, ax



out_dir = 'results/2021-03-19_00:16:09_relative_gap_EMA_mid'
gs = [0.5,1,1.5,2,2.5,3]

objs = zload(out_dir + '/objectives.pkl')
obj0, objILP, objfluid, objfluidInt = objs
fig, ax  = plot_diff_gs(gs, obj0, objILP, objfluid, objfluidInt)
plt.savefig(out_dir+'/relative_gap.pdf')


out_dir = 'results/2021-03-18_23:37:04_sparse_LA_EMA_mid'
objs = zload(out_dir + '/objectives.pkl')
objs = [i/objs[0] for i in objs]
max_lanes_vec = zload(out_dir + '/max_lanes_vec.pkl')
fig, ax = plt.subplots(figsize=(4,2))
plt.plot(max_lanes_vec, objs, marker='.')
plt.xlabel('Max. number of reversals, $k$')
plt.ylabel('$J^{\\text{LP}}_k/J^{\\text{Original}}$')
plt.xlim([0, max(max_lanes_vec)])
plt.tight_layout()
plt.savefig(out_dir+'/sparse_LA.pdf')


#out_dir = 'results/'+dir_out +'_'+ net_name


