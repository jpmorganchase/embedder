###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright 2025: Amazon Web Services, Inc. - Contributions from JPMC
###############################################################################
"""ahs_utils_demo.py: Helper logic for Analog Hamiltonian Simulation (AHS)."""

import time
import numpy as np
import networkx as nx
from collections import Counter
from braket.devices import LocalSimulator
from braket.ahs.atom_arrangement import AtomArrangement


def generate_graph(atom_positions, scale=4.0*1e-6):
    """
    Helper function to generate a NetworkX graph and Braket AtomArrangement,
    with union-jack (UJ) connectivity, given specified parameters for size of 
    underlying square lattice, and
    atomic positions given as list of tuples [(0,0), (0,1), ...].

    Input:
        atom_positions: Positions of nodes (atoms) in 2D square lattice
        scale: [Optional] Lattice spacing a in SI units (defaults to 4um)
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    lattice_width = max([x for x,_ in atom_positions])+1
    lattice_height = max([y for _,y in atom_positions])+1
    atom_positions_si = [(x*scale,y*scale) for x,y in atom_positions]
    # how to label the nodes (uncomment one)
    # continuous numbers [0...n-1]
    node_labels = range(len(atom_positions))
    # node at x,y has the label x + lattice_width*y
    #node_labels = [y*lattice_width + x for x,y in atom_positions]

    edge_dict = {}
    for i in range(len(atom_positions)):
        x, y = atom_positions[i]
        edge_dict[node_labels[i]] = []
        for j in range(i+1,len(atom_positions)):
            u, v = atom_positions[j]
            if abs(x-u) <= 1 and abs(y-v) <=1:
                edge_dict[node_labels[i]] += [node_labels[j]]

    graph = nx.from_dict_of_lists(edge_dict)

    atoms = AtomArrangement()
    for atom in atom_positions_si:
        atoms.add(atom)
    
    return atoms, graph


def run_local(ahs_program, shots=1000):
    "Helper function to run experiment with local AHS simulator"

    # run program on classical device for simulation (without noise)
    device_id = "braket_ahs"
    device = LocalSimulator(device_id)
    nshots = shots

    start = time.time()
    task = device.run(ahs_program, shots=nshots)
    stop = time.time()

    # The result can be downloaded directly into an object in the python session:
    result = task.result()
    runtime = stop-start

    print(f"Time to run AHS with local simulator (in seconds): {runtime}")

    return result, runtime


def get_result_dic(result):
    '''
    Helper function to get mesurement result as dictionary
    '''
    result_dict = {"measurements":[]}
    for measurement in result.measurements:
        shot_result = {
            "pre_sequence":[int(qubit) for qubit in measurement.pre_sequence],
            "post_sequence":[int(qubit) for qubit in measurement.post_sequence]
                      } 
        result_dict["measurements"].append(shot_result)
    return result_dict


def get_counts(result_dic):
    """Helper function to aggregate occurence counts"""

    # convert notation to e (empty), r (rydberg) and g (ground state
    states = ['e', 'r', 'g']
    state_labels = []
    for shot in result_dic['measurements']:
        pre = shot['pre_sequence']
        post = shot['post_sequence']
        state_idx = np.array(pre) * (1 + np.array(post))
        state_labels.append("".join([states[s_idx] for s_idx in state_idx]))
    
    # get occurence counts
    occurence_count = Counter(state_labels)
    
    return state_labels, occurence_count