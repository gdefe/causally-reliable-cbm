from networkx import nodes
import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.trainer import Trainer
from src.hydra_parsing import parse_hyperparams, target_classname
from src.data.utils import static_graph_collate
from src.metrics import edge_type
from torch.distributions import MultivariateNormal
import scipy
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.optimize import minimize as minimize_scipy
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import Bounds, NonlinearConstraint
import warnings
import numbers
import random

def model_has_concepts(model):
    if target_classname(model) in ['BlackBox_Multi', 'CBM', 'CEM', 'C2BM', 'SCBM']:
        return True
    elif target_classname(model) in ['BlackBox']:
        return False
    else:
        raise ValueError(f"Unknown model type: {target_classname(model)}")
    

def model_is_causal(model):
    if target_classname(model) in ['C2BM']:
        return True
    elif target_classname(model) in ['BlackBox', 'BlackBox_Multi', 'CEM', 'CBM', 'SCBM']:
        return False
    else:
        raise ValueError(f"Unknown model type: {target_classname(model)}")

def clean_empty_configs(cfg: DictConfig) -> DictConfig:
    """ can be used to set default values for missing keys """
    with open_dict(cfg):
        if not cfg.get('causal_discovery'):
            cfg.update(causal_discovery = None)
        if not cfg.get('llm'):
            cfg.update(llm = None)
        if not cfg.get('rag'):
            cfg.update(rag = None)
    return cfg

# def update_config_from_model(cfg: DictConfig) -> DictConfig:
#     """ can be used to update the config based on the model """
#     if not model_is_causal(cfg.model):
#         cfg.causal_discovery = None
#     return cfg

def update_config_from_data(cfg: DictConfig, dataset) -> DictConfig:
    """ can be used to update the config based on the data, e.g., set input and output size """
    with open_dict(cfg):
        cfg.engine.model.update(
            input_size = dataset.data["train"].X.shape[-1] if dataset.data["train"].X is not None else None,
            output_size = dataset.y_info['cardinality'][0], # we assume single class classification
            c_info = dataset.c_info,
            y_info = dataset.y_info,
        )
        cfg.engine.update(
            c_names = dataset.c_info['names']
        )
    return cfg

def maybe_update_config_with_graph(cfg: DictConfig, graph, interv_policy) -> DictConfig:
    """ can be used to update the config based on the graph """
    if graph is not None:
        if model_is_causal(cfg.model):
            with open_dict(cfg):
                cfg.engine.model.update(
                    graph = graph.values.tolist(),
                    graph_labels = graph.index.tolist()
                )
    if interv_policy:
        with open_dict(cfg):
            cfg.engine.update(
                test_interv_policy = interv_policy
            )
    return cfg
    
def finetune_model(cfg, engine, dataset):
    ftune_dataloader = DataLoader(dataset.data['ftune'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate) 
    ftuneval_dataloader = DataLoader(dataset.data['ftune_val'], batch_size=cfg.dataset.batch_size, collate_fn=static_graph_collate)
    # Freeze all parameters except encoder
    for name, param in engine.model.named_parameters():
        param.requires_grad = 'encoder' in name
    trainer = Trainer(cfg)
    trainer.logger.log_hyperparams(parse_hyperparams(cfg))
    trainer.fit(engine, ftune_dataloader, ftuneval_dataloader)
    return trainer, engine

def get_parents(graph, i):
    # get the indices of the parents of the node i
    return graph[:,i].nonzero().squeeze(1)

def get_roots(graph):
    return graph.sum(dim=0) == 0


def dfs(node, adj_matrix, visited, stack, remove):
    visited[node] = True
    stack[node] = True
    for neighbor in range(len(adj_matrix)):
        if adj_matrix[neighbor][node] == 1:
            if not visited[neighbor]:
                if dfs(neighbor, adj_matrix, visited, stack, remove):
                    return True
            elif stack[neighbor]:
                if remove:
                    adj_matrix[neighbor][node] = 0
                    print(f'The cycle has been broken by removing the edge: {neighbor} -> {node}')
                return True
    stack[node] = False
    return False

def contains_cycle(adj_matrix):
    visited = [False] * len(adj_matrix)
    stack = [False] * len(adj_matrix)
    for node in range(len(adj_matrix)):
        if not visited[node]:
            if dfs(node, adj_matrix, visited, stack, False):
                return True
    return False

def remove_cycles(graph, start_node):
    """
    This function removes the cycles in the graph by removing the last visited edges
    before the cycle is detected
    Args:
        graph (Dataframe): the adjacency matrix of the graph
        start_node: the index of the task node
    Returns:
        graph (Dataframe): the adjacency matrix of the graph without cycles
    """
    adj_matrix = graph.values
    if contains_cycle(adj_matrix):
        while contains_cycle(adj_matrix):
            visited = [False] * len(adj_matrix)
            stack = [False] * len(adj_matrix)
            dfs(start_node, adj_matrix, visited, stack, True)
    else:
        print('there are no cycles in the graph, therefore the graph is left untouched')
    graph = pd.DataFrame(adj_matrix, index=graph.index, columns=graph.columns, dtype=int)
    return graph

def remove_problematic_edges(graph, dataset):
    graph, virtual_c_names = common_cause_nodes(graph)
    if virtual_c_names:
        for split in dataset.data:
            dataset.data[split].c = torch.cat([torch.full((len(dataset.data[split].c), len(virtual_c_names)), float('nan')), 
                                            dataset.data[split].c], 
                                            dim=1)
        dataset.c_info['names'] = virtual_c_names + dataset.c_info['names']
        dataset.c_info['cardinality'] = [2]*len(virtual_c_names) + dataset.c_info['cardinality']
    else:
        print('therefore no virtual nodes where added and the ground truth graph and C are left untouched') 
    return graph, dataset

def get_graph_levels(graph, task_node):
    # extract the subgraph of the task node
    involeved_nodes, roots = get_task_graph(graph, task_node)
    # get the levels of the graph
    levels = get_levels(graph, involeved_nodes, roots)
    # check if the levels and roots are correct
    check_graph(levels, graph)
    return levels

def common_cause_nodes(graph):
    """
    This function removes the bi-directed and undirected edges in the graph
    and adds virtual nodes (roots) to model the common cause
    Args:
        adj_mat: the adjacency matrix of the graph
    Returns:
        adj_mat: the adjacency matrix of the graph with virtual
    """
    adj_mat = graph.values
    virtual_node_names = []
    pairs = []

    for i in range(len(adj_mat)):
        for j in range(i, len(adj_mat)):
            e_t = edge_type(adj_mat, i, j)
            if e_t in ['i<->j', 'i-j']: # if edge is undirected or bi-directed
                raise ValueError(f'The {e_t} edge between {i} and {j} is either undirected or bi-directed, therefore the LLM-step is needed to proceed with a DAG.')
                # adj_mat[i, j] = 0 # the edge is removed
                # adj_mat[j, i] = 0
                # print(f'The {e_t} edge between {i} and {j} has been removed')
                # pairs.append((i, j))
    
    for cnt, pair in enumerate(pairs):
        i = pair[0]+cnt
        j = pair[1]+cnt
        vector = np.zeros(len(adj_mat)); vector[i] = 1; vector[j] = 1
        adj_mat = np.row_stack([vector, adj_mat])
        adj_mat = np.column_stack([np.zeros(len(adj_mat)), adj_mat]) 
        virtual_node_names.append(f'#virtual_{cnt}') # add virtual node name
        print(f'The virtual node {virtual_node_names[-1]} has been added')


    if len(virtual_node_names) == 0:
        print('no undirected or bidirected edges where found')
    else:
        labels = virtual_node_names + list(graph.index)
        graph = pd.DataFrame(adj_mat, index=labels, columns=labels, dtype=int)
    return graph, virtual_node_names


def get_task_graph(graph, task_node):
    # start from the last in the graph, i.e., the index of the task
    branches = [[task_node]]
    nodes = [task_node]
    roots = []
    while True:
        parents = [get_parents(graph, node) for node in nodes]
        parents = torch.unique(torch.cat(parents)).tolist()
        # if parents are roots, add them later
        roots_to_add = [p for p in parents if len(get_parents(graph, p))==0]
        parents = [p for p in parents if p not in roots_to_add]
        roots = roots + roots_to_add
        if len(parents) == 0:
            break
        # remove nodes from previous branches if they are neessary at this step
        branches = [[node for node in level if node not in parents] for level in branches]
        # add nodes to the level
        branches.append(parents)
        nodes = parents
    roots = list(set(roots))
    involved_nodes = [node for branch in branches for node in branch]
    return involved_nodes, roots

def get_levels(graph, involved_nodes, roots):
    # start from the top of the graph
    # 1st level: get all nodes that requires only the roots to be computed
    # 2nd level: then get all nodes that only requires the root and the previous nodes
    # and so on...
    previous_level = roots
    levels = []
    while len(involved_nodes) > 0:
        level = [node for node in involved_nodes if set(get_parents(graph, node).tolist()).issubset(previous_level)]
        levels.append(level)
        previous_level = previous_level + level
        involved_nodes = [node for node in involved_nodes if node not in previous_level]
    levels.insert(0, roots)
    return levels

def check_graph(graph_levels, true_graph):
    roots = graph_levels[0].copy()
    levels = graph_levels[1:].copy()

    # check roots found are a subset of the total roots
    all_roots = torch.where(get_roots(true_graph))[0].tolist()
    assert set(roots).issubset(all_roots), \
        "The roots found are not a subset of the total roots"

    levels.insert(0, roots)
    # check all nodes appear only ones in the graph levels
    involved_nodes = [node for branch in levels for node in branch]
    assert [involved_nodes.count(node) for node in involved_nodes] == [1]*len(involved_nodes), \
        "Some nodes appear more than once in the graph levels"

    for level in levels:
        if level == roots:
            continue
        for node in level:
            parents = get_parents(true_graph, node).tolist()
            # check parents appear before their children in the graph levels
            for parent in parents:
                node_index = [levels.index(l) for l in levels if node in l]; assert len(node_index) == 1; node_index = node_index[0]
                parent_index = [levels.index(l) for l in levels if parent in l]; assert len(parent_index) == 1; parent_index = parent_index[0]
                assert node_index > parent_index, \
                f"Parent {parent} appear after their children {node} in the graph level {level}"

            # check at least one parent is in the previous level
            assert any([p in levels[levels.index(level)-1] for p in parents]), \
                f"At least one parent of {node} should be in the previous level"

            # check if the position of the nodes in the graph levels correspond to
            # the number of edges to a root
            node_index = [levels.index(l) for l in levels if node in l]; assert len(node_index) == 1; node_index = node_index[0]
            len_node_to_roots = 0
            while True:
                len_node_to_roots += 1
                if set(parents).issubset(set(roots)):
                    break
                parents = [get_parents(true_graph, parent) for parent in parents]
                parents = torch.unique(torch.cat(parents)).tolist()
            assert node_index == len_node_to_roots, \
                f"The position of the node {node} in the graph levels is not correct"

def maybe_corrupt_graph(graph, corruption_type = 'random_perturbation', corrupt_graph_percentage = 0.1):
    """
    Corrupt the graph by randomly removing/ adding/ flipping edges.
    Attributes:
        graph (Dataframe): A dictionary containing the graph information.

    Returns:
        Dataframe: The corrupted graph.
    """
    corrupted_matrix = graph.values
    n = len(corrupted_matrix)
    print('Corrupting causal graph...')

    if corruption_type == 'random_perturbation':
        # randomly add, remove, or flip a percentage of all possible edges in the graph
        
        upper_indices = [(i, j) for i in range(n-1) for j in range(i+1, n)]
        # shuffle the upper indices to randomly select edges
        random.shuffle(upper_indices)

        k = int(corrupt_graph_percentage * len(upper_indices))  # number of changes to make: a percentage of the total number of edges
        if k == 0:
            print('No changes to be made, k is 0')
            return graph

        print(f'Number of changes to be made: {k}')
        changes = 0
        idx = 0
        while changes < k and idx < len(upper_indices):
            i, j = upper_indices[idx]
            a, b = corrupted_matrix[i, j], corrupted_matrix[j, i]
            changes += 1
            idx += 1

            if (a, b) == (1, 0):  # edge i → j exists
                op = random.choice(["remove", "flip"])
                if j == n-1:
                    # if the edge is to the last node, we can only remove it
                    # otherise the task node will have a child
                    op = "remove"
                    if np.sum(corrupted_matrix[:, -1] == 1) <= 1:
                        print("The last node has no incoming edges, therefore we cannot remove the edge", i, "->", j)
                        changes -= 1
                        continue  # don't remove the last edge to the final node
                       
                if op == "remove":
                    print("Removing edge:", i, "->", j)
                    corrupted_matrix[i, j] = 0
                    corrupted_matrix[j, i] = 0
                elif op == "flip":
                    print("Flipping edge:", i, "->", j, "to", i, "<-", j)
                    corrupted_matrix[i, j] = 0
                    corrupted_matrix[j, i] = 1

            elif (a, b) == (0, 1):  # edge j → i exists
                op = random.choice(["remove", "flip"])
                if op == "remove":
                    print("Removing edge:", j, "->", i)
                    corrupted_matrix[i, j] = 0
                    corrupted_matrix[j, i] = 0
                elif op == "flip":
                    print("Flipping edge:", j, "->", i, "to", j, "<-", i)
                    corrupted_matrix[i, j] = 1
                    corrupted_matrix[j, i] = 0
            elif (a, b) == (0, 0):  # no edge exists, we can add one of the two
                op = random.choice(["direct", "indirect"])
                if j == n-1:
                    op = "direct"

                if op == "direct":
                    print("Adding edge:", i, "->", j)
                    corrupted_matrix[i, j] = 1
                    corrupted_matrix[j, i] = 0
                else:
                    print("Adding edge:", j, "->", i)
                    corrupted_matrix[i, j] = 0
                    corrupted_matrix[j, i] = 1
            else:
                print("Unexpected edge type between ", i, "and ", j)
                changes -= 1 
                continue  # ignore ambiguous or inconsistent cases, if exists (e.g., -1)



    elif corruption_type == 'cbm_perturbation':

        # randomly connecting nodes to the task node
        # exclude the last node (task node)
        nodes_notask = list(range(n-1))
        random.shuffle(nodes_notask)

        k = max(1, int(len(nodes_notask) * corrupt_graph_percentage))
        
        if k == 0:
            print('No changes to be made, k is 0')
            return graph

        print(f'Number of nodes to connect to the task: {k}')
        for i in nodes_notask[:k]:
            print("Connecting node ", i, "to the task node and removing other outgoing edges")
            # If there are outgoing edges to nodes other than the task node, remove them
            corrupted_matrix[i,:(n-1)]=0
            # add the edge i -> n-1 (task node)
            corrupted_matrix[i, n-1] = 1
               

    elif corruption_type == 'random_graph':
    
        corrupted_matrix = np.zeros((n, n), dtype=int)
        k = int(corrupt_graph_percentage * n * (n - 1)/2)  # number of edges to add
        if k == 0:
            print('No edge to be inserted, k is 0')
            return graph

        print(f'Number of edges to be added: {k}')
        while k > 0:
            i = random.randint(0, n-2)
            j = random.randint(i+1, n-1)
            if k == 1 and np.sum(corrupted_matrix[:, -1] == 1) == 0:  
                # if the last node has no incoming edges, we can only add an edge to it
                j = n-1  
            op = random.choice(["direct", "indirect"])
            if j == n-1:  # if the edge is to the last node, we can only add it
                op = "direct"

            if corrupted_matrix[i, j] == 0 and corrupted_matrix[j, i] == 0:  # no edge exists
                k -= 1
                if op == "direct":  # add edge i -> j
                    corrupted_matrix[i, j] = 1
                    print(f'Adding edge: {i} -> {j}')
                else:  # add edge j -> i
                    corrupted_matrix[j, i] = 1             
                    print(f'Adding edge: {j} -> {i}')

    # check if the graph has cycles and remove them
    corrupted_graph = pd.DataFrame(corrupted_matrix, index=graph.index, columns=graph.columns, dtype=int)
    corrupted_graph = remove_cycles(corrupted_graph, len(corrupted_graph) - 1)  # fix the graph after corruption   
    print("Done corrupting the graph")
    return corrupted_graph

def get_policy_from_graph(graph, y_index):
    # get the levels of the graph
    torch_values_graph = torch.tensor(graph.values)
    policy = get_graph_levels(torch_values_graph, y_index)
    # remove the task node
    assert policy[-1][0] == y_index
    policy = policy[:-1]
    # get the names of the nodes
    names = list(graph.index)
    level_names = [[names[i] for i in level] for level in policy]
    return policy, level_names

def get_intervention_policy(type, pred_graph, true_graph, y_index):
    assert true_graph is not None or pred_graph is not None, 'Intervention policy: a graph is required for any policy.'
    names = list(pred_graph.index) if pred_graph is not None else list(true_graph.index)

    if type == 'random':
        policy = np.random.choice(np.arange(len(names)), size=len(names), replace=False)
        policy = [[int(i)] for i in policy]
        policy = [level for level in policy if level[0] != y_index]
    elif type == 'all':
        policy = np.arange(len(names))
        policy = [[int(i)] for i in policy]
        policy = [level for level in policy if level[0] != y_index]
    elif type == 'levels_true' or type == 'nodes_true':
        if true_graph is not None:
            policy, level_names = get_policy_from_graph(true_graph, y_index)
        else:
            print('Intervention policy: no true graph found, using the predicted graph.')
            policy, level_names = get_policy_from_graph(pred_graph, y_index)
    elif type == 'levels_pred' or type == 'nodes_pred':
        if pred_graph is not None:
            policy, level_names = get_policy_from_graph(pred_graph, y_index)
        else:
            raise ValueError('Intervention policy: no predicted graph found.')
    else:
        raise ValueError('Intervention policy: unknown policy type.')
    
    if type == 'nodes_true' or type == 'nodes_pred':
        policy = [[node] for level in policy for node in level]

    # filter virtual roots from intervention policy
    policy = [[node for node in level if '#virtual_' not in names[node]] for level in policy]
    level_names = [[names[i] for i in level] for level in policy] 
    return policy, level_names

def numerical_stability_check(cov, device, epsilon=1e-6):
    """
    Check for numerical stability of covariance matrix.
    If not stable (i.e., not positive definite), add epsilon to diagonal.

    Parameters:
    cov (Tensor): The covariance matrix to check.
    epsilon (float, optional): The value to add to the diagonal if the matrix is not positive definite. Default is 1e-6.

    Returns:
    Tensor: The potentially adjusted covariance matrix.
    """
    num_added = 0
    if cov.dim() == 2:
        cov = (cov + cov.transpose(dim0=0, dim1=1)) / 2
    else:
        cov = (cov + cov.transpose(dim0=1, dim1=2)) / 2

    while True:
        try:
            # Attempt Cholesky decomposition; if it fails, the matrix is not positive definite
            torch.linalg.cholesky(cov)
            if num_added > 0.0001:
                print(
                    "Added {} to the diagonal of the covariance matrix.".format(
                        num_added
                    )
                )
            break
        except RuntimeError:
            # Add epsilon to the diagonal
            if cov.dim() == 2:
                cov = cov + epsilon * torch.eye(cov.size(0), device=device)
            else:
                cov = cov + epsilon * torch.eye(cov.size(1), device=device)
            num_added += epsilon
            epsilon *= 2
    return cov

class SCBMPercentileStrategy:
    # Set intervened concept logits to 0.05 & 0.95
    def __init__(self):
        pass

    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        c_intervened_probs = (0.05 + 0.9 * c_true) * c_mask
        c_intervened_logits = torch.logit(c_intervened_probs, eps=1e-6)
        return c_intervened_logits