import numpy as np
import scipy as sp
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import normalize
from collections import Counter


# ##########################
# Kernel Matrix functions #
###########################

def nh_kernel_matrix(graph_set, R=1):
    """ compute the kernel matrix of a set of graphs using the NHK and label
    comparison """

    N = len(graph_set)
    K_set = []

    computation_size = R * (N ** 2 - sum(range(N + 1)))
    print("Total number of graphs: {0}".format(N))
    print("Total number of graph comparisons: {0}".format(computation_size))

    for r in range(R):

        #compute neighbor hash for nodes in every graph
        for i in tqdm(N, desc="Computing NH"):
            graph_set[i] = neighborhood_hash(graph_set[i])

        #precompute the label histogram for each graph
        graph_set_hist = []
        for i in tqdm(N, desc="Computing Label Hist"):
            g = graph_set[i]
            hist = label_histogram(g)
            graph_set_hist.append(hist)

        #compute upper triangular kernel matrix
        K = np.identity(N)
        for i in tqdm(N, desc="Computing K"):
            for j in range(i + 1, N):
                k = histogram_intersection(graph_set_hist[i],
                                           graph_set_hist[j])
                K[i, j] = k
        #build lower triangle
        K = K + K.transpose() - np.identity(len(K))
        np.savez_compressed("K_{0}.npz".format(r), K)

        K_set.append(K)

    #normalization of K
    return sum(K_set) / len(K_set)


# TODO
# def csnh_kernel_matrix(graph_set, R=1):


def nh_explicit_data_matrix(graph_set, R=1):
    """ Compute the data matrix of graphs after applying the neighborhood hash.
    Every feature vector is a histogram of labels in the hashed graph.
    """

    N = len(graph_set)
    print("Total number of graphs: {0}".format(N))

    for r in range(R):
        #compute neighbor hash for nodes in every graph
        for i in tqdm(N, desc="Computing NH"):
            graph_set[i] = neighborhood_hash(graph_set[i])

    #compute all feature vectors from histograms as sparse 0,N matrices
    X = []
    for i in tqdm(N, desc="Computing X"):
        g = graph_set[i]
        h = label_histogram(g)
        X.append(h)
    X = np.vstack(X)

    return X


def csnh_explicit_data_matrix(graph_set, R=1):
    """ Compute the data matrix of graphs after applying the cost-sensitive
        neighborhood hash. Every feature vector is a histogram of labels in
        the hashed graph.
    """

    N = len(graph_set)
    print("Total number of graphs: {0}".format(N))

    for r in range(R):
        # compute cost sensitive neighbor hash for nodes in every graph
        print("Starting iteration {0}...".format(r))
        for i in tqdm(N, desc='Computing CSNH'):
            graph_set[i] = count_sensitive_neighborhood_hash(graph_set[i])

    #compute all feature vectors from histograms as sparse 0,N matrices
    X = []
    for i in tqdm(N, desc='Computing X'):
        g = graph_set[i]
        h = label_histogram(g)
        X.append(h)
    X = np.vstack(X)

    return X


def simple_node_hash_data_matrix(graph_set):
    """ Compute the data matrix of graphs after applying the simple node hash.
    Every feature vector is a histogram of labels in the original graph.
    """

    N = len(graph_set)
    print("Total number of graphs: {0}".format(N))

    #compute all feature vectors from histograms as sparse 0,N matrices
    X = []
    for i in tqdm(N, desc="Computing X"):
        g = graph_set[i]
        h = label_histogram(g)
        X.append(h)
    X = np.vstack(X)

    return X


def xor_neighborhood_hash_data_matrix(graph_set, R=1):
    """ Compute the data matrix of graphs after applying an XOR
        neighborhood hash. Every feature vector is a histogram of labels in
        the hashed graph.
    """

    N = len(graph_set)
    print("Total number of graphs: {0}".format(N))

    for r in range(R):

        #compute cost sensitive neighbor hash for nodes in every graph 
        print("Starting iteration {0}...".format(r))
        for i in tqdm(N, desc='Computing XORNH'):
            graph_set[i] = xor_neighborhood_hash(graph_set[i])

    #compute all feature vectors from histograms as sparse 0,N matrices
    X = []
    for i in tqdm(N, desc="Computing X"):
        g = graph_set[i]
        h = label_histogram(g)
        X.append(h)
    X = np.vstack(X)

    return X


####################
# Kernel functions #
####################

def neighborhood_hash_kernel(g1, g2):
    """ Compute the NH kernel of two graphs as described by Hido et al.
        in "A Linear-time Graph Kernel (2009)"
    """

    g1_nh_hist = label_histogram(neighborhood_hash(g1))
    g2_nh_hist = label_histogram(neighborhood_hash(g2))
    k = histogram_intersection(g1_nh_hist, g2_nh_hist)

    return k


def count_sensitive_neighborhood_hash_kernel(g1, g2):
    """ Compute the Count-Sensitive NH kernel of two graphs as described
        by Hido et al. in "A Linear-time Graph Kernel (2009)"
    """

    g1_csnh_hist = label_histogram(count_sensitive_neighborhood_hash(g1))
    g2_csnh_hist = label_histogram(count_sensitive_neighborhood_hash(g2))
    k = histogram_intersection(g1_csnh_hist, g2_csnh_hist)

    return k


def random_walk_kernel(g1, g2, parameter_lambda, node_attribute='label'):
    """ Compute the random walk kernel of two graphs as described by Neuhaus
        and Bunke in the chapter 5.9 of "Bridging the Gap Between Graph Edit
        Distance and Kernel Machines (2007)"
    """
    p = nx.cartesian_product(g1, g2)
    M = nx.attr_sparse_matrix(p, node_attr=node_attribute)
    A = M[0]
    L = A.shape[0]
    k = 0
    A_exp = A
    for n in range(L):
        # k += (parameter_lambda ** n) * long(A_exp.sum()) ##/////////Orignal/////////////////////
        k += (parameter_lambda ** n) * int(A_exp.sum()) ##///////////Changed b/c long not present in Python3///
        if n < L:
            A_exp = A_exp * A

    return k



#################################
# Auxiliary Functions on Graphs #
#################################

def neighborhood_hash(g):
    """ Compute the simple neighborhood hashed version of a graph.
    """
    gnh = g ##/////////////////Changed/////////
    # gnh = g.copy() ##//////////Orignal/////////

    for node in iter(g.nodes()):
        # neighbors_labels = [g.node[n]["label"] for n in g.neighbors_iter(node)] ##//////Orignal//////
        neighbors_labels = [g.node[n]["label"] for n in g.neighbors(node)] ##////////Changed b/c nx 2.0///////
        # print("Nodes=", g.nodes())
        # print("neighbors Label=", type(neighbors_labels))
        if len(neighbors_labels) > 1:
            x = neighbors_labels[0]
            # for n in g.neighbors(node):
            #     print(n, g.node[n]["label"])
            # print(g.neighbors(node), neighbors_labels)
            for i in neighbors_labels[1:]:
                x = np.bitwise_xor(x, i) ##/////////////Error because one label only//////////
            node_label = g.node[node]["label"]
            nh = np.bitwise_xor(np.roll(node_label, 1), x)
        else:
            nh = g.node[node]["label"]

        gnh.node[node]["label"] = nh

    return gnh


def count_sensitive_neighborhood_hash(g):
    """ Compute the count sensitive neighborhood hashed
        version of a graph.
    """

    gnh = g.copy()
    g = array_labels_to_str(g)

    #iterate over every node in the graph
    for node in iter(g.nodes()):
        # neighbors_labels = [g.node[n]["label"] for n in g.neighbors_iter(node)] ##//////Orignal//////
        neighbors_labels = [g.node[n]["label"] for n in g.neighbors(node)]  ##////////Changed b/c nx 2.0///////

        #if node has no neighbors, nh is its own label
        if len(neighbors_labels) > 0:

            #count number of unique labels
            c = Counter(neighbors_labels)
            count_weighted_neighbors_labels = []
            # for label, c in c.iteritems(): ##///////////Orignal/////////////////////
            for label, c in c.items(): ##/////////////Changed b/c dict in python3
                label = str_to_array(label)
                c_bin = np.array(list(np.binary_repr(c, len(label))),
                                 dtype=np.int64)
                label = np.bitwise_xor(label, c_bin)
                label = np.roll(label, c)
                count_weighted_neighbors_labels.append(label)
            x = count_weighted_neighbors_labels[0]
            for l in count_weighted_neighbors_labels[1:]:
                x = np.bitwise_xor(x, l)
            node_label = str_to_array(g.node[node]["label"])
            csnh = np.bitwise_xor(np.roll(node_label, 1), x)
        else:
            csnh = str_to_array(g.node[node]["label"])

        gnh.node[node]["label"] = csnh

    return gnh


def xor_neighborhood_hash(g):
    """ Compute the xor neighborhood hashed version of a graph.
    """

    gnh = g.copy()

    for node in iter(g.nodes()):
        # neighbors_labels = [g.node[n]["label"] for n in g.neighbors_iter(node)] ##/////Orignal///////
        neighbors_labels = [g.node[n]["label"] for n in g.neighbors(node)] ##///////Changed b/c nx 2.0
        if len(neighbors_labels) > 0:
            l = g.node[node]["label"]
            for i in neighbors_labels:
                l = np.bitwise_xor(l, i)
            nh = l
        else:
            nh = g.node[node]["label"]

        gnh.node[node]["label"] = nh

    return gnh



def array_labels_to_str(g):
    """ convert all binary array labels to strings in a graph """

    for n in g.node.items():
        n[1]['label'] = ''.join([str(l) for l in n[1]['label']])
    return g


def str_labels_to_array(g):
    """ convert all string labels to binary arrays in a graph """

    for n in g.node.items():
        n[1]['label'] = np.array(list(n[1]['label']), dtype=np.int64)
    return g


def label_histogram(g):
    """ Compute the histogram of labels in nx graph g. Every label is a
        binary array. The histogram length is 2**len(label)
    """

    labels = [g.node[name]["label"] for name in g.nodes()]
    # print(labels, len(labels))
    h = np.zeros(2 ** len(labels[0]))
    # print("H=",h, len(h))
    for l in labels:
        h[int(''.join([str(i) for i in l]), base=2)] += 1
    return h


def neighborhood_sizes(g):
    # return [len(g.neighbors(node)) + 1 for node in iter(g.nodes())] ##/////////Orignal//////////////////////
    return [len(list(g.neighbors(node))) + 1 for node in iter(g.nodes())] ##/////Changed/////////////////////


def neighborhood_size_distribution(g):
    """ Returns a counter of the sizes of neighborhoods
        in a graph. This is useful to observe the distribution
        of length-1 substructures sizes.
    """
    # neighborhood_sizes = [len(g.neighbors(node)) + 1 for node in iter(g.nodes())] ##////////Orignal///////////
    neighborhood_sizes = [len(list(g.neighbors(node))) + 1 for node in iter(g.nodes())] ##//////////Changed//////////
    return Counter(neighborhood_sizes)


###################################
# Auxiliary Functions on Matrices #
###################################

def make_binary(X):
    """ Transforms every element in X in a binary vector of n ones and m-n zeros
    where n is the value of the element and m is the maximum element in X.

    Args:
        X: a (N,M) matrix or array

    Returns:
        X: a (N,M * m) binary matrix
    """

    N, M = X.shape
    # print(X.shape)
    m = np.max(X)
    print(X.shape, N, M, m)
    X_bin = sp.sparse.lil_matrix((N, M * m), dtype=np.int8)

    for i in tqdm(range(N), desc="Making X binary"): ##//////////////Changed because only one iteration///
    # for i in tqdm(N, desc="Making X binary"): ##/////////Original//////////////////////
        for j in range(M):
            n = X[i, j]
            # print(n)
            X_bin[i, m * j:(m * j + n)] = 1 ## 1,10285*1:(
    X_bin = make_sparse(X_bin)#sp.sparse.csr_matrix(X_bin)
    return X_bin, m


def make_binary_bounded(X):
    """ Sets an element-wise bound to 1. If an element is larger than 0, it set
    to 1.

    Args:
        X: a matrix or (N,M) array

    Returns:
        X: a binary matrix or (N,M) array
    """
    X[X > 0] = 1
    return X


def make_sparse(X):
    return sp.sparse.csr_matrix(X)


def normalize_matrix(X):
    return normalize(make_sparse(X), norm='l1', axis=1, copy=False)


def histogram_intersection(h1, h2):
    """ Compute the minimum common number of elements in two histograms c
        and normalize c using the total number of elements in the histograms
    """

    c = sum(np.minimum(h1, h2))
    k = c / (sum(h1) + sum(h2) - c)
    return k


def array_to_str(a):
    return ''.join([str(i) for i in a])


def str_to_array(s):
    return np.array(list(s), dtype=np.int64)
