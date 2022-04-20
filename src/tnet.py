import networkx as nx
import pandas as pd
from gurobipy import *
from src.utils import *
import numpy as np
import src.msa as msa
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.linalg import block_diag
import copy
import time
from pyproj import Proj
import re 

class tNet():
    def __init__(self, netFile=None, gFile=None, G=None, g=None, fcoeffs=[1, 0, 0, 0, 0.15, 0]):
        """
        Initialize a Traffic Network object. It requieres networkx and Gurobi libraries

        Parameters
        ----------
        self : a tNet object

        netFile: a network file in the format proposed by Bar Gera

        gFile: an OD demand file in the format proposed by Bar Gera

        Returns
        -------
        A tNet Object.

        """

        if G == None and netFile != None:
            G = readNetFile(netFile)
        elif G != None and netFile == None:
            G = G

        node_id_map = {k: G.nodes[k]['node name'] for k in G.nodes()}
        if g == None and gFile != None:
            g = readODdemandFile(gFile, node_id_map)
        elif g != None and gFile == None:
            g = g

        self.netFileName = netFile
        self.gFileName = gFile
        self.nLinks = len(G.edges())
        self.nNodes = len(G.nodes())
        self.nOD = len(g)
        self.Zones = getZones(g)
        self.nZones = getNumZones(g)
        self.totalDemand = sum(g.values())
        self.g = g
        self.gGraph = buildDemandGraph(g)
        self.G = G
        self.node_id_map = node_id_map
        self.fcoeffs = fcoeffs
        self.nPoly = len(fcoeffs)
        #self.TAP = self.build_TAP(self.G)
        self.incidence_matrix, self.link_id_dict = incidence_matrix(self.G)

    def build_TAP(self, G , n_iter=5000):
        """
	    Build a traffic assignment object based on the traffic network 
	    Jurgen Hackl <hackl@ibi.baug.ethz.ch>

	    Parameters
	    ----------

		gdict: OD demands dict

	    Returns
	    -------
	    An nx object.

	    """
        assert (self.nZones > 0), "Number of Zones is zero!!"
        assert (self.totalDemand > 0), "Total demand is zero!!"
        assert (self.nNodes > 0), "No nodes in graph!!"
        assert (self.nLinks > 0), "No links in graph!!"
        TAP = msa.TrafficAssignment(G, self.gGraph,
                                    fcoeffs=self.fcoeffs,
                                    iterations=n_iter,
                                    threshold=0.0025)
        return TAP

    def build_pedestrian_net(self):
        """
        build a pedestrian net by considering both directions links

        Parameters
        ----------

        self: a tnet object

        Returns
        -------
        An attribute on the object containing the pedestrian network

        """
        G = nx.DiGraph()
        for i, j in self.G.edges():
            G.add_edge(i, j, length=self.G[i][j]['length'])
            G.add_edge(j, i, length=self.G[i][j]['length'])
        self.G_pedestrian = G


    def build_layer(self, one_way=True, avg_speed=3.1 ,capacity=99999, symb="'", identical_G=False):
        """
        build aditional layer, one or two ways with avg speed and capacity

        Parameters
        ----------

        self: a tnet object

        Returns
        -------
        updated supergraph

        """
        G2 = self.G_supergraph.copy()
        if identical_G == False:
            if one_way == True:
                [G2.add_edge(str(i) + symb, str(j) + symb, length=self.G[i][j]['length'], t_0=self.G[i][j]['length'] / avg_speed, capacity=capacity, type=symb) for i, j in self.G.edges() if isinstance(i, int) == True and isinstance(j, int)]

                #[for i, j in self.G.edges() if isinstance(i, int) == True and isinstance(j, int)]
                #for i, j in self.G.edges() :
                #    if isinstance(i, int) == True and isinstance(j, int):
                 #       G2.add_edge(str(i) + symb, str(j) + symb, length=self.G[i][j]['length'], t_0=self.G[i][j]['length'] / avg_speed, capacity=capacity, type=symb)
                 #       G2.add_edge(i, str(i) + symb, t_0=0.1, capacity=capacity, type='f', length=0.1)
                 #       G2.add_edge(str(i) + symb, i, t_0=0.1, capacity=capacity, type='f', length=0.1)
                 #       G2.add_edge(j, str(j) + symb, t_0=0.1, capacity=capacity, type='f', length=0.1)
                 #       G2.add_edge(str(j) + symb, j, t_0=0.1, capacity=capacity, type='f', length=0.1)
            else:
                [G2.add_edge(str(i) + symb, str(j) + symb, length=self.G[i][j]['length'], t_0=self.G[i][j]['length'] / avg_speed, capacity=capacity, type=symb) for i, j in self.G.edges() if isinstance(i, int) == True and isinstance(j, int)]
                [G2.add_edge(str(j) + symb, str(i) + symb, length=self.G[i][j]['length'], t_0=self.G[i][j]['length'] / avg_speed, capacity=capacity, type=symb) for i, j in self.G.edges() if isinstance(i, int) == True and isinstance(j, int)]
                #for i, j in self.G.edges():

                #    if isinstance(i, int) == True and isinstance(j, int):
                #        G2.add_edge(str(i) + symb, str(j) + symb, length=self.G[i][j]['length'],
                #                   t_0=self.G[i][j]['length'] / avg_speed, capacity=capacity, type=symb)
                #        G2.add_edge(str(j) + symb, str(i) + symb, length=self.G[i][j]['length'],
                 #                  t_0=self.G[i][j]['length'] / avg_speed, capacity=capacity, type=symb)
                 #       G2.add_edge(i, str(i) + symb, t_0=0.1, capacity=capacity, type='f', length=0.1)
                 #       G2.add_edge(str(i) + symb, i, t_0=0.1, capacity=capacity, type='f', length=0.1)
                 #       G2.add_edge(j, str(j) + symb, t_0=0.1, capacity=capacity, type='f', length=0.1)
                 #       G2.add_edge(str(j) + symb, j, t_0=0.1, capacity=capacity, type='f', length=0.1)
            [G2.add_edge(i, str(i) + symb, t_0=10/60/60, capacity=capacity, type='f', length=0.1) for i, j in self.G.edges() if isinstance(i, int) == True and isinstance(j, int)]
            [G2.add_edge(str(i) + symb, i, t_0=10/60/60, capacity=capacity, type='f', length=0.1) for i, j in self.G.edges() if isinstance(i, int) == True and isinstance(j, int)]
            [G2.add_edge(j, str(j) + symb, t_0=10/60/60, capacity=capacity, type='f', length=0.1) for i, j in self.G.edges() if isinstance(i, int) == True and isinstance(j, int)]
            [G2.add_edge(str(j) + symb, j, t_0=10/60/60, capacity=capacity, type='f', length=0.1) for i, j in self.G.edges() if isinstance(i, int) == True and isinstance(j, int)]
        self.G_supergraph = G2


    def build_supergraph(self, walk_multiplier=1, identical_G=False):
        """
        build a supergraph mixing pedestrian and vehicle networks

        Parameters
        ----------

        self: a tnet object

        Returns
        -------
        An attribute on the object containing the pedestrian network

        """
        G2 = copy.deepcopy(self.G)#.copy()
        if identical_G == False:
            for i, j in self.G.edges():
                if isinstance(i, int) == True and isinstance(j, int):
                    G2.add_edge(str(i) + "'", str(j) + "'", length=self.G[i][j]['length'],
                               t_0=self.G[i][j]['length'] / 3.1 / walk_multiplier, capacity=10000, type='p')
                    G2.add_edge(str(j) + "'", str(i) + "'", length=self.G[i][j]['length'],
                               t_0=self.G[i][j]['length'] / 3.1 / walk_multiplier, capacity=10000, type='p')
                    G2.add_edge(i, str(i) + "'", t_0=10/60/60, capacity=99999, type='f', length=0.1)
                    G2.add_edge(str(i) + "'", i, t_0=10/60/60, capacity=99999, type='f', length=0.1)
                    G2.add_edge(j, str(j) + "'", t_0=10/60/60, capacity=99999, type='f', length=0.1)
                    G2.add_edge(str(j) + "'", j, t_0=10/60/60, capacity=99999, type='f', length=0.1)
        self.G_supergraph = G2

    def add_walking_layer(self, walk_multiplier=1):
        self.build_supergraph(walk_multiplier=walk_multiplier)

    def add_layer(self, layer, layer_symb, speed=False):
        """
        add a new layer to the supergraph

        Parameters
        ----------

        self: a tnet object

        Returns
        -------
        Updates the G_supergraph object and returns it

        """
        G2 = self.G_supergraph.copy()
        if speed == False:
            for i,j in layer.edges():
                G2.add_edge(str(i) + layer_symb, str(j) + layer_symb, length=layer[i][j]['length'],
                           t_0=layer[i][j]['t_0'], capacity=layer[i][j]['capacity'], type=layer_symb)
                G2.add_edge(i, str(i) + layer_symb, t_0=10/60/60, capacity=99999, type='f', length=0.1)
                G2.add_edge(str(i) + layer_symb, i, t_0=10/60/60, capacity=99999, type='f', length=0.1)
                G2.add_edge(j, str(j) + layer_symb, t_0=10/60/60, capacity=99999, type='f', length=0.1)
                G2.add_edge(str(j) + layer_symb, j, t_0=10/60/60, capacity=99999, type='f', length=0.1)
        else:
            for i, j in layer.edges():
                G2.add_edge(str(i) + layer_symb, str(j) + layer_symb, length=layer[i][j]['length'],
                            t_0=layer[i][j]['t_0']/speed, capacity=layer[i][j]['capacity'], type=layer_symb)
                G2.add_edge(i, str(i) + layer_symb, t_0=10/60/60, capacity=99999, type='f', length=0.1)
                G2.add_edge(str(i) + layer_symb, i, t_0=10/60/60, capacity=99999, type='f', length=0.1)
                G2.add_edge(j, str(j) + layer_symb, t_0=10/60/60, capacity=99999, type='f', length=0.1)
                G2.add_edge(str(j) + layer_symb, j, t_0=10/60/60, capacity=99999, type='f', length=0.1)
        self.G_supergraph = G2
        return self.G_supergraph



    def solveMSA(self, exogenous_G=False):
        """
	    Solve the MSA flows for a traffic network using the MSA module by 
	    Jurgen Hackl <hackl@ibi.baug.ethz.ch>

	    Parameters
	    ----------

		gdict: OD demands dict

	    Returns
	    -------
	    An nx object.

	    """
        t0 = time.process_time()
        self.TAP = self.build_TAP(self.G, n_iter=300)
        self.TAP.run(fcoeffs=self.fcoeffs, build_t0=False, exogenous_G=exogenous_G)
        runtime = time.process_time() - t0
        self.G = self.TAP.graph
        return runtime, self.TAP.RG

    def solveMSA_social(self, exogenous_G=False):
        """
        Solve the MSA flows for a traffic network using the MSA module by
        Jurgen Hackl <hackl@ibi.baug.ethz.ch>

        Parameters
        ----------

        gdict: OD demands dict

        Returns
        -------
        An nx object.

        """
        t0 = time.process_time()
        self.TAP = self.build_TAP(self.G, n_iter=600)
        self.TAP.run_social(fcoeffs=self.fcoeffs, build_t0=False, exogenous_G=exogenous_G)
        runtime = time.process_time() - t0
        self.G = self.TAP.graph
        return runtime, self.TAP.RG

    def solveMSAsocial_supergraph(self, build_t0=False, exogenous_G=False):
        """
        Solve the MSA social flows for a traffic network using the MSA module by
        Jurgen Hackl <hackl@ibi.baug.ethz.ch>

        Parameters
        ----------

        gdict: OD demands dict

        Returns
        -------
        An nx object.

        """

        self.TAP = self.build_TAP(self.G_supergraph)
        t0 = time.process_time()
        TT = self.TAP.run_social(fcoeffs=self.fcoeffs, build_t0=build_t0, exogenous_G=exogenous_G)
        runtime = time.process_time() - t0
        self.G_supergraph = self.TAP.graph

        return TT, runtime

    def solveMSAsocial_capacity_supergraph(self, build_t0=False, exogenous_G=False, d_step=1, n_iter=200):
        """
        Solve the MSA social flows for a traffic network using the MSA module by
        Jurgen Hackl <hackl@ibi.baug.ethz.ch>

        Parameters
        ----------

        gdict: OD demands dict

        Returns
        -------
        An nx object.

        """
        self.TAP = self.build_TAP(self.G_supergraph, n_iter=n_iter)
        t0 = time.process_time()
        TT, d_norm, RG = self.TAP.run_social_capacity(fcoeffs=self.fcoeffs, build_t0=build_t0, exogenous_G=exogenous_G, d_step=d_step)
        runtime = time.process_time() - t0
        self.G_supergraph = self.TAP.graph
        return TT[1:], d_norm[1:], runtime, RG

    def set_fcoeffs(self, fcoeff):
        """
	    set function coefficients of the transportation netwrok object

	    Parameters
	    ----------

		fcoeffs: function coefficients of a polynomial function

	    Returns
	    -------
	    updates the fcoeffs attribute of the tnet object

	    """
        self.fcoeffs = fcoeff

    # TAP = msa.TrafficAssignment(self.G, self.gGraph, fcoeffs=fcoeff)

    def set_g(self, g):
        """
	    sets a demand dictionary and updates the the tNet and TAP objects

	    Parameters
	    ----------

		g: an OD demand dictionary

	    Returns
	    -------
	    updates the objects in the tnet where g takes place

	    """
        self.nOD = len(g)
        self.Zones = getZones(g)
        self.nZones = getNumZones(g)
        self.totalDemand = sum(g.values())
        self.g = g
        self.gGraph = buildDemandGraph(g)
        #self.TAP = self.build_TAP(self.G)

    def read_flow_file(self, fname):
        """
	    Read flow file in tntp format. If G is provided, 
	    we add the info to G, otherwise, generate a nx object.

	    Parameters
	    ----------
		
		fname: input file name
	    G: networkx obj

	    Returns
	    -------
	    a networkx object 

	    """
        read_flowFile(fname, self.G)

    def write_flow_file(self, fname):
        """
	    Write flow file in tntp format. If G is provided, 
	    we add the info to G, otherwise, generate a nx object.

	    Parameters
	    ----------
		
		fname: input file name
	    G: networkx obj

	    Returns
	    -------
	    a networkx object 

	    """
        write_flowFile(self.G, fname)

    def travel_time(self, i, j):
        """
        evalute the travel time function for edge i->j

        Parameters
        ----------

        i: starting node of edge
        j: ending node of edge

        Returns
        -------
        float

        """
        return sum([self.fcoeffs[n] * (self.G[i][j]['flow'] / self.G[i][j]['capacity']) ** n for n in
                    range(len(self.fcoeffs))])

    def eval_obj(self):
        """
        evalute total travel time in network

        Parameters
        ----------
        self

        Returns
        -------
        float

        """
        return sum([self.G[i][j]['flow'] * self.G[i][j]['t_0'] * self.travel_time(i, j) for i, j in self.G.edges()])

    def read_node_coordinates(self, fname):
        """
        reads node coordinates to the graph network

        Parameters
        ----------
        self
        fname: file containing the node coordinates

        Returns
        -------
        modified object

        """
        with open(fname) as file:
            file_lines = file.readlines()

        od_d = {}
        od_d_t = {}
        for line in file_lines:
            if "x" in line:
                continue
            if ";" in line:
                line_ = line.split()
                n = int(line_[0])
                x = float(line_[1])
                y = float(line_[2])
                self.G.nodes[n]['pos'] = (x, y)

    def set_supergraph_tk(self, fcoeffs, G_exogenous=False):
        """
        Set travel times

        Parameters
        ----------

        G: networkx obj

        Returns
        -------
        a float

        """
        for i, j in self.G_supergraph.edges():
            self.G_supergraph[i][j]['t_k'] = travel_time(self.G_supergraph, fcoeffs, i, j, G_exogenous=False)

    
    def latlong2xy(self):
        for n in self.G.nodes():
            self.G.nodes[n]['pos'] = latlong_to_xy(self.G.nodes[n]['pos']) 
        for n in self.G_supergraph.nodes():
            n_ = int(re.sub("[^0-9]", "", str(n)))
            if int(n_) >= 99999:
                self.G_supergraph.nodes[n]['pos'] = self.G.nodes[n2]['pos']    
            else:
                n2 = int(re.sub("[^0-9]", "", str(n)))
                self.G_supergraph.nodes[n]['pos'] = self.G.nodes[n2]['pos']


    def OD_to_tex(self):

        max_s = max([s for s,t in self.g.keys()])
        max_t = max([t for s,t in self.g.keys()])
        M = np.zeros((max_t, max_s))
        for w,d in self.g.items():
            M[w[0]-1][w[1]-1] = d
        df = pd.DataFrame(M)
        df.columns += 1
        df.index += 1
        d = df.to_latex(float_format="%.0f",
                        column_format = 'c|'+'c'*len(df.columns))
        print(d)

def latlong_to_xy(pos):
    pp = Proj(proj='utm',zone=10,ellps='WGS84', preserve_units=False)
    return pp(pos[0],pos[1])
       



def plot_network_flows(G, weight='flow', width=3, cmap=plt.cm.Blues):
    # TODO: add explaination
    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(G, 'pos')
    edges, weights = zip(*nx.get_edge_attributes(G, weight).items())
    nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=width, edge_cmap=cmap)
    labels = {(i, j): int(G[i][j][weight]) for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    return fig, ax

def plot_network(G, ax, edgelist=None, edge_width=0.1,
	edgecolors='k', nodecolors='lightgray', nodesize=1, arrowstyle='fancy', arrowsize=0.1, edge_alpha=1, linkstyle='-', connectstyle='arc3, rad=0.05', cmap=plt.cm.inferno_r, vmin=0, vmax=1):
    if edgelist == None:
        edgelist = list(G.edges())
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos,
                edgelist=edgelist,
                width=edge_width,
                ax=ax,
                edge_color=edgecolors,
                node_size=nodesize,
                node_color=nodecolors,
                linewidths=8,
                connectionstyle=connectstyle,
                #arrowstyle=None,
                arrowsize=arrowsize,
                alpha=edge_alpha,
                style=linkstyle,
                edge_cmap=cmap,
                edge_vmin=vmin,
            edge_vmax=vmax,
            with_labels=True)
    #else:
    #    nx.draw(G, pos, width=edge_width,
    #            ax=ax, edge_color=edgecolors,
    #            node_size=nodesize, node_color=nodecolors,
    #            #connectionstyle='arc3, rad=0.04',
    #            arrowsize=0.5, arrowstyle='fancy',
    #            alpha=edge_alpha, style=linkstyle)
'''
def plot_network(G, width=1, cmap=plt.cm.Blues):
    # TODO: add explaination
    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(G, 'pos')
    #edges = list(G.edges())
    #nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=width, edge_cmap=cmap)
    nx.draw(G, pos, node_color='k',  width=width, edge_cmap=cmap, arrowsize=1, node_size=1, alpha=0.7,
        connectionstyle='arc3, rad=0.04'
            )
    #labels = {(i, j): int(G[i][j][weight]) for i, j in G.edges()}
    #nx.draw_networkx_edge_labels(G, pos=pos, with_labels=True)
    return fig, ax
'''
def perturbDemandConstant(g, constant):
    """
    Perturb demand by a random constant value

    Parameters
    ----------

	g: demand dict
	max_var: maximum percentage of the demands which is goinf to variate either
			increase or decrease

    Returns
    -------
    a perturbed demand dict

    """
    a = {}
    for od, demand in g.items():
        a[od] = demand * constant
    return a


def incidence_matrix(G):
    """
    build incidence matrix and column index dictionary. Note that since 
    node 0 don't exist, the node is the column number plus one

    Parameters
    ----------

	a nx element

    Returns
    -------
    A sparse matrix and a dictionary to match line number and link

    """
    nNodes = len(G.nodes())
    nLinks = len(G.edges())
    N = np.zeros((nNodes, nLinks))
    link_dict = {}
    idx = 0
    for s, t in G.edges():
        link_dict[idx] = (s, t)
        N[s - 1, idx] = -1
        N[t - 1, idx] = 1
        idx += 1
    return N, link_dict


def buildDemandGraph(g):
    """
    a nx graph defininf OD demands

    Parameters
    ----------

	gdict: OD demands dict

    Returns
    -------
    An nx object.

    """
    od_graph = nx.DiGraph()
    for (s, t), d in g.items():
        od_graph.add_edge(s, t, demand=d)
    return od_graph


def readNetFile(netFile, sep="\t"):
    """
    Read the netFile and convert it to a nx object

    Parameters
    ----------

    netFile: a network file in the format proposed by Bar Gera

    Returns
    -------
    An nx object.

    """

    # Create a networkx obj
    G = nx.DiGraph()
    # Read the network file
    with open(netFile) as file_flow:
        file_flow_lines = file_flow.readlines()

    for line in file_flow_lines:
        if ";" in line and "~" not in line:
            links = line.split(sep)
     #       if 'Subway' in netFile:
     #           print(links[1]+'->'+links[2])
            G.add_edge(int(links[1]), int(links[2]), capacity=float(links[3]), \
                       length=float(links[4]), t_0=float(links[5]), \
                       B=float(links[6]), power=float(links[7]), speedLimit=float(links[8]), \
                       toll=float(links[9]), type=float(links[10]))
    #G = nx.convert_node_labels_to_integers(G, first_label=1, ordering='sorted', label_attribute='node name')
    #print(list(G.nodes()))
    for n in G.nodes():
        G.nodes[n]['node name'] = n

    return G


def readODdemandFile(gfile, node_id_map):
    """
	Read the gfile and convert it to a dict

	Parameters
	----------

	gFile: a demand file in the format proposed by Bar Gera

	Returns
	-------
	A dict with OD as key and demand as value

	"""

    with open(gfile) as trips:
        trip_lines = trips.readlines()

    od_d = {}
    od_d_t = {}
    for line in trip_lines:
        if "Origin" in line:
            origin = node_id_map[int(line.split("gin")[1])]
        if ";" in line:
            line_ = line.split(";")
            for j in line_:
                if ":" in j:
                    dest = node_id_map[int(j.split(":")[0])]
                    d = float(j.split(":")[1])
                    if d == 0:
                        continue
                    if origin != dest:
                        od_d[(origin, dest)] = d
    return od_d


@timeit
def writeNetfile(G, g, fname):
    """
	write net file from G (networkx)

	Parameters
	----------

	G: a neteworkx object

	Returns
	-------
	file
	"""
    nZones = str(len(set([i for i, j in g.keys()])))
    nNodes = str(len(G.nodes()))
    nLinks = str(len(G.edges()))
    header = "<NUMBER OF ZONES> " + nZones + "\n<NUMBER OF NODES> " + nNodes + "\n<FIRST THRU NODE> 1\n<NUMBER OF LINKS> " + nLinks + "\n<END OF METADATA>\n~  Init node  Term node  Capacity  Length  Free Flow Time  B  Power  Speed limit  Toll  Type  ;\n"
    text = ""
    idx = 0
    link_id = {}
    for (s, t) in G.edges():
        idx += 1
        link_id[idx] = (s, t)
        text += "\t" + str(s) + "\t" + str(t) + "\t" + str(G[s][t]["capacity"]) + "\t" + str(
            G[s][t]["length"]) + "\t" + str(G[s][t]["t_0"]) \
                + "\t" + str(G[s][t]["B"]) + "\t" + str(G[s][t]["power"]) + "\t" + str(
            G[s][t]["speedLimit"]) + "\t" + str(G[s][t]["toll"]) \
                + "\t" + str(G[s][t]["type"]) + "\t;\n"
    write_file(header + text, fname)
    return link_id, header + text

@timeit
def writeNetfile2(G,  fname):
    """
	write net file from G (networkx)

	Parameters
	----------

	G: a neteworkx object

	Returns
	-------
	file
	"""
    nZones = ''
    nNodes = str(len(G.nodes()))
    nLinks = str(len(G.edges()))
    header = "<NUMBER OF ZONES> " + nZones + "\n<NUMBER OF NODES> " + nNodes + "\n<FIRST THRU NODE> 1\n<NUMBER OF LINKS> " + nLinks + "\n<END OF METADATA>\n~  Init node  Term node  Capacity  Length  Free Flow Time  B  Power  Speed limit  Toll  Type  ;\n"
    text = ""
    idx = 0
    link_id = {}
    for (s, t) in G.edges():
        idx += 1
        link_id[idx] = (s, t)
        text += "\t" + str(s) + "\t" + str(t) + "\t" + str(G[s][t]["capacity"]) + "\t" + str(
            G[s][t]["length"]) + "\t" + str(G[s][t]["t_0"]) \
                + "\t" + str(G[s][t]["B"]) + "\t" + str(G[s][t]["power"]) + "\t" + str(
            G[s][t]["speedLimit"]) + "\t" + str(G[s][t]["toll"]) \
                + "\t" + str(G[s][t]["type"]) + "\t;\n"
    write_file(header + text, fname)
    return link_id, header + text


@timeit
def writeTripsfile(g, fname):
    """
	write trips file from dict 

	Parameters
	----------
]
	g dict

	Returns
	-------
	file
	"""
    nZones = str(len(set([i for i, j in g.keys()])))
    totalFlow = sum([d for d in g.values()])
    header = "<NUMBER OF ZONES> " + nZones + "\n<TOTAL OD FLOW> " + str(totalFlow) + "\n<END OF METADATA>\n\n"

    text = ""
    nodes = list(set([s for s, t in g.keys()]))
    nodes.extend(list(set([t for s, t in g.keys()])))
    nodes = list(set(nodes))
    for o in nodes:
        txt = ""
        txt = "Origin " + str(o) + "\n"
        demandTxt = ""
        for d in nodes:
            try:
                gk = str(g[(o, d)])
            except:
                gk = str(0)
            demandTxt += str(d) + "\t : \t" + str(gk) + ";\n"
        text += txt + demandTxt + "\n\n"
    write_file(header + text, fname)
    return header + text


def getZones(gdict):
    """
    Returns Zones in a OD file

    Parameters
    ----------

    gdict: a demand dictiornary

    Returns
    -------
    a list with the Zones of the network

    """
    sources = [s for (s, t) in gdict.keys()]
    targets = [t for (s, t) in gdict.keys()]
    sources.extend(targets)
    return set(sources)


def getNumZones(gdict):
    """
    Finds the number of zones in a network

    Parameters
    ----------

    gdict: a demand dictiornary

    Returns
    -------
    number of zones

    """
    return len(getZones(gdict))


def greenshieldFlow(speed, capacity, free_flow_speed):
    """
    Returns the flow of a link following the Fundamental Diagram (Greenshield's Model).

    Parameters
    ----------

    speed: miles/hr or km/hr
    capacity: link's capacity in veh/hr
    free_flow_speed: link's free flow speed in miles/hr or km/hr

    Returns
    -------
    flow: resulting flow 

    """
    if speed > free_flow_speed or capacity < 0:
        return 0
    x = 4 * capacity * speed / free_flow_speed - 4 * capacity * (speed ** 2) / (free_flow_speed ** 2)
    return x


def travel_time(G, fcoeffs, i, j, G_exogenous=False):
    """
	evalute the travel time function for edge i->j

	Parameters
	----------
	tnet: transportation network object
	i: starting node of edge
	j: ending node of edge

	Returns
	-------
	float

	"""
    if G_exogenous == False:
        return sum([fcoeffs[n] * (G[i][j]['flow'] / G[i][j]['capacity']) ** n for n in range(len(fcoeffs))])
    else:
        return sum([fcoeffs[n] * ((G[i][j]['flow'] + G_exogenous[i][j]['flow']) / G[i][j]['capacity']) ** n for n in
                    range(len(fcoeffs))])


def get_totalTravelTime(G, fcoeffs, G_exogenous=False):
    """
    Return the total travel time of a network

    Parameters
    ----------

    G: networkx obj

    Returns
    -------
    a float 

    """
    return sum([G[i][j]['flow'] * G[i][j]['t_0'] * travel_time(G, fcoeffs, i, j, G_exogenous=G_exogenous) for i, j in
                G.edges()])




def read_flowFile(fname, G=False):
    """
	Read flow file in tntp format. If G is provided, 
	we add the info to G, otherwise, generate a nx object.

	Parameters
	----------

	fname: input file name
	G: networkx obj

	Returns
	-------
	a networkx object 

	"""
    with open(fname) as flows:
        flow_lines = flows.readlines()
    if G == False:
        G = nx.DiGraph()
    for line in flow_lines:
        if "Tail" not in line and ";" in line:
            line_ = line.split("\t")
            i = 0
            for j in line_:
                if i == 1:
                    s = int(j)
                elif i == 2:
                    t = int(j)
                elif i == 3:
                    flow = float(j)
                i += 1
            if G == False:
                G.add_edge(s, t)
                G[s][t]['flow'] = flow
            else:
                G[s][t]['flow'] = flow
    return G


def write_flowFile(G, fname):
    """
    Write flow file in tntp format. If G is provided, 
    we add the info to G, otherwise, generate a nx object.

    Parameters
    ----------
	
	fname: input file name
    G: networkx obj

    Returns
    -------
    a networkx object 

    """
    nNodes = str(len(G.nodes()))
    nLinks = str(len(G.edges()))
    header = "~\tTail\tHead\tVolume\t;\n"
    text = ""
    idx = 0
    link_id = {}
    for (s, t) in G.edges():
        idx += 1
        link_id[idx] = (s, t)
        text += "\t" + str(s) + "\t" + str(t) + "\t" + str(G[s][t]["flow"]) + "\t;\n"
    write_file(header + text, fname)


def add_net_flows(array):
    # TODO: add description

    G = array[0].G.copy()
    for tn in array[1:]:
        for i, j in G.edges():
            G[i][j]['flow'] += tn.G[i][j]['flow']
    return G


def add_G_flows(array):
    # TODO: add description

    G = array[0].copy()
    for tn in array[1:]:
        for i, j in G.edges():
            G[i][j]['flow'] += tn[i][j]['flow']
    return G


def flowDiff(graph1, graph2):
    """
    Return the difference between the flows of two graphs.

    Parameters
    ----------

    graph1: netwrorkx obj
    graph2: networkx obj

    Returns
    -------
    a dictionary with name od edge and diff in flows. graph1-graph2

    """
    flowDiff_ = {}
    for edge in graph1.edges():
        flow1 = graph1[edge[0]][edge[1]]['flow']
        flow2 = graph2[edge[0]][edge[1]]['flow']
        flowDiff_[edge] = flow1-flow2
    return flowDiff_

def normFlowDiff(G_data, G_estimate):
    """
    estimate the difference between the flows of two networks
    with identical network topology.

    Parameters
    ----------

    G_data: nx network with edge attribute "flow"
    G_estimate: nx network with edge attribute "flow"

    Returns
    -------
    a scalar representing the norm of the flows of two netorks

    """
    diff = sum([(G_data.get_edge_data(s,t)['flow'] - G_estimate.get_edge_data(s,t)['flow'])**2 for s,t in G_data.edges()])
    return diff


def get_total_G_flow(G):
    """
    get total flow in a graph

    Parameters
    ----------

    G: networkx graph with edge attribute Flow

    Returns
    -------
    float

    """
    return sum([G[i][j]['flow'] for i,j in G.edges()])


from datetime import datetime

def get_network_parameters(net_name, experiment_name='_'):
    if net_name == 'Braess1':
        netFile = "data/net/Braess1_net.txt"
        gFile = "data/trips/Braess1_trips.txt"
        fcoeffs = [1, 1, 0, 0, 0., 0]
    elif net_name == 'Braess2':
        netFile = "data/net/Braess2_net.txt"
        gFile = "data/trips/Braess2_trips.txt"
        fcoeffs = [1, 1, 0.001, 0., 0., 0.]
    elif net_name == 'EMA':
        netFile = "data/net/EMA_net.txt"
        gFile = "data/trips/EMA_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'EMA_mid':
        netFile = "data/net/EMA_mid_net.txt"
        gFile = "data/trips/EMA_mid_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'NYC_small':
        netFile = "data/net/NYC_small_net.txt"
        gFile = "data/trips/NYC_small_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
        #fcoeffs  = [1, 0.07363975723108054, 0.049250179014950504, 0.04848249228476019, 0.12321501861763541, 0.017914889834778975]
    elif net_name == 'NYC_Uber_small':
        netFile = "data/net/NYC_Uber_small_net.txt"
        gFile = "data/trips/NYC_Uber_small_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'NYC_Uber_small_1':
        netFile = "data/net/NYC_Uber_small_net.txt"
        gFile = "data/trips/hola2.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
        #fcoeffs = [1, 0.07363975723108054, 0.049250179014950504, 0.04848249228476019, 0.12321501861763541,
        #           0.017914889834778975]
    elif net_name == 'berlin-tiergarten':
        netFile = "data/net/berlin-tiergarten_net.txt"
        gFile = "data/trips/berlin-tiergarten_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'Anaheim':
        netFile = "data/net/Anaheim_net.txt"
        gFile = "data/trips/Anaheim_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'Sioux Falls':
        netFile = "data/net/SiouxFalls_net.txt"
        gFile = "data/trips/SiouxFalls_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'Winnipeg':
        netFile = "data/net/Winnipeg_net.txt"
        gFile = "data/trips/Winnipeg_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'Barcelona':
        netFile = "data/net/Barcelona_net.txt"
        gFile = "data/trips/Barcelona_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'Sydney':
        netFile = "data/net/Sydney_net.txt"
        gFile = "data/trips/Sydney_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'SiouxFalls':
        netFile = "data/net/SiouxFalls_net.txt"
        gFile = "data/trips/SiouxFalls_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'ChicagoSketch':
        netFile = "data/net/ChicagoSketch_net.txt"
        gFile = "data/trips/ChicagoSketch_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    elif net_name == 'test_9':
        netFile = "data/net/test_9_net.txt"
        gFile = "data/trips/test_9_trips.txt"
        fcoeffs = [1, 0, 0, 0, 0.15, 0]
    tstamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dir_out = tstamp + "_"+ experiment_name

    return netFile, gFile, fcoeffs, tstamp, dir_out


def read_net(net_name):
    if net_name == 'NYC':
        tNet, tstamp, fcoeffs = nyc.build_NYC_net('data/net/NYC/', only_road=True)
    else:
        netFile, gFile, fcoeffs, tstamp, dir_out = get_network_parameters(net_name=net_name,
                                                                               experiment_name=net_name + '_n_variation')
        print(netFile)
        net = tNet(netFile=netFile, gFile=gFile, fcoeffs=fcoeffs)
    return net, fcoeffs