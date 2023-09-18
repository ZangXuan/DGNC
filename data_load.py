# -*-coding:utf-8 -*-

import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp
import pickle
import pandas as pd
import random

def save_nx_graph(nx_graph, path):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file

def load_any_obj_pkl(path):
    ''' load any object from pickle file
    '''
    with open('./data/' + path + '.pkl', 'rb') as f:
        any_obj = pickle.load(f)
    return any_obj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def create_graphs(file,num_nodes,t_span):
    graphs = []
    df = pd.read_csv(file + '.txt', sep=' \t| ', names=['from', 'to', 'time'], header=None, comment='%')
    time_ini = min(df['time'])
    print(time_ini)
    print(max(df['time']))
    num_snapshot = (max(df['time']) - time_ini) // t_span + 1
    print(num_snapshot)
    for i in range(num_snapshot):
        g = nx.Graph()
        time_start = i * t_span + int(time_ini)
        time_end = (i + 1) * t_span + int(time_ini)
        for j in range(num_nodes):
            g.add_node(j)
        for j in range(len(df['time'])):
            if time_start <= df['time'][j] < time_end:
                g.add_edge(int(df['from'][j]), int(df['to'][j]),timestamp=df['time'][j])

        graphs.append(g)
        print('processed graphs ', i, '/','ALL done......')
        print(len(g.nodes()), "   ", len(g.edges()))

    save_nx_graph(nx_graph=graphs, path=file + '.pkl')


def load_labels(file,num_nodes):
    labels = np.empty(shape=(num_nodes, 1))
    labels_dic = {}
    labels_num = 0
    with open('./data/' + file + '_labels.txt', 'r', encoding='utf-8') as filein:
        for line in filein:
            line_list = line.strip('\n').split(' ')  # 我这里的数据之间是以 ; 间隔的
            if line_list[1] not in labels_dic:
                labels_dic[line_list[1]] = labels_num
                labels_num += 1
            labels[int(line_list[0])] = labels_dic[line_list[1]]
    print('The number of labels: ',len(labels_dic))
    return labels

def load_profile(file,t_span):
    graphs = load_any_obj_pkl(file)
    adj_list = []
    timestamp_list = []
    onehot = sp.eye(graphs[0].number_of_nodes())
    onehot = sparse_mx_to_torch_sparse_tensor(onehot)
    for i in range(len(graphs)):
        adj_now = nx.adjacency_matrix(graphs[i], weight=None)
        adj_now = sparse_mx_to_torch_sparse_tensor(adj_now)
        adj_list.append(adj_now)

        timestamp_now = torch.zeros(graphs[i].number_of_nodes(), graphs[i].number_of_nodes())

        for e in graphs[i].edges:
            timestamp_now[e[0],e[1]] = int(graphs[i].edges[e[0], e[1]]['timestamp']) - i * t_span
        timestamp_list.append(timestamp_now)

    return adj_list, timestamp_list, onehot, graphs

def neg_sample(adj,num_nodes):
    neg_list = []
    for a in adj:
        a = a.to_dense()
        num = torch.sum(a)
        temp = 0
        neg = torch.zeros(num_nodes,num_nodes)
        while temp<num:
            m = np.random.choice(num_nodes)
            n = np.random.choice(num_nodes)
            if m!=n and a[m][n]==0:
                neg[m][n] += 1
                temp += 1
            else:
                continue
        neg_list.append(neg)
    return neg_list
