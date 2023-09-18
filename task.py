import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import random
import statistics
import math
from sklearn.cluster import KMeans

def test_sample(adj):
    a = adj.to_dense()
    num = torch.sum(a)
    num_nodes = a.size()[0]
    temp = 0
    pos = torch.nonzero(a, as_tuple=False)
    neg = []
    while temp < num:
        m = np.random.choice(num_nodes)
        n = np.random.choice(num_nodes)
        if m != n and a[m][n] == 0:
            neg.append([m,n])
            temp += 1
        else:
            continue
    random.shuffle(pos)
    random.shuffle(neg)
    return pos,neg

def get_link_score(fu, fv):
    """Given a pair of embeddings, compute link feature based on operator (such as Hadammad product, etc.)"""
    fu = np.array(fu)
    fv = np.array(fv)
    return np.multiply(fu, fv)

def get_link_feats(links, emb):
    """Compute link features for a list of pairs"""
    emb = emb.cpu().detach().numpy()
    features = []
    for l in links:
        a, b = l[0], l[1]
        f = get_link_score(emb[a,:], emb[b,:])
        features.append(f)
    return features

def link_prediction(emb, adj, val_mask_fraction=0.2,test_mask_fraction=0.6):
    pos, neg = test_sample(adj)
    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(pos, neg,
                                                                            test_size=val_mask_fraction + test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos, test_neg,
                                                                                    test_size=test_mask_fraction / (
                                                                                            test_mask_fraction + val_mask_fraction))

    train_pos_feats = get_link_feats(train_edges_pos, emb)
    train_neg_feats = get_link_feats(train_edges_neg, emb)
    test_pos_feats = get_link_feats(test_edges_pos, emb)
    test_neg_feats = get_link_feats(test_edges_neg, emb)
    val_pos_feats = get_link_feats(val_edges_pos, emb)
    val_neg_feats = get_link_feats(val_edges_neg, emb)

    train_pos_labels = np.array([1] * len(train_pos_feats))
    train_neg_labels = np.array([-1] * len(train_neg_feats))
    val_pos_labels = np.array([1] * len(val_pos_feats))
    val_neg_labels = np.array([-1] * len(val_neg_feats))
    test_pos_labels = np.array([1] * len(test_pos_feats))
    test_neg_labels = np.array([-1] * len(test_neg_feats))

    train_data = np.vstack((train_pos_feats, train_neg_feats))
    train_labels = np.append(train_pos_labels, train_neg_labels)
    val_data = np.vstack((val_pos_feats, val_neg_feats))
    val_labels = np.append(val_pos_labels, val_neg_labels)
    test_data = np.vstack((test_pos_feats, test_neg_feats))
    test_labels = np.append(test_pos_labels, test_neg_labels)

    logistic = linear_model.LogisticRegression(solver='lbfgs')
    logistic.fit(train_data, train_labels)
    test_predict = logistic.predict_proba(test_data)[:, 1]
    val_predict = logistic.predict_proba(val_data)[:, 1]

    test_roc_score = roc_auc_score(test_labels, test_predict)
    val_roc_score = roc_auc_score(val_labels, val_predict)

    test_AP_score = average_precision_score(test_labels, test_predict)
    val_AP_score = average_precision_score(val_labels, val_predict)

    #print('val_AUC', val_roc_score, 'test_AUC', test_roc_score, 'val_AP',val_AP_score, 'test_AP', test_AP_score)

    return val_roc_score,test_roc_score,val_AP_score,test_AP_score


def classification_sample(num_nodes):
    random.seed(0)
    train_ratios = [0.3, 0.5, 0.7]
    datas = []
    for ratio in train_ratios:
        data = []
        for i in range(40):
            idx_val = random.sample(range(num_nodes), int(num_nodes * 0.2))
            remaining = np.setdiff1d(np.array(range(num_nodes)), idx_val)
            idx_train = random.sample(list(remaining), int(num_nodes * ratio))
            idx_test = np.setdiff1d(np.array(remaining), idx_train)
            data.append([idx_train, idx_val, list(idx_test)])
        datas.append(data)
    return datas


def node_classification(emb, labels, num_nodes):
    datas = classification_sample(num_nodes)
    average_accs = []
    average_aucs = []
    macro_f1 = []
    micro_f1 = []
    for train_nodes in datas:
        temp_accs = []
        temp_aucs = []
        temp_macro_f1 = []
        temp_micro_f1 = []
        for train_node in train_nodes:
            train_vec = emb.cpu().detach().numpy()[train_node[0]]
            train_y = labels[train_node[0]]
            val_vec = emb[train_node[1]]
            val_y = labels[train_node[1]]
            test_vec = emb.cpu().detach().numpy()[train_node[2]]
            test_y = labels[train_node[2]]
            test_vec_auc = torch.softmax(emb, dim=1).cpu().detach().numpy()[train_node[2]]

            clf = linear_model.LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=4000)
            # clf = linear_model.LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=200)
            clf.fit(train_vec, train_y)

            y_pred = clf.predict_proba(test_vec)
            # print('y_pred',y_pred)
            # print('test_y',test_y)
            y_true = np.zeros_like(y_pred)
            for i in range(y_true.shape[0]):
                try:
                    y_true[i, int(test_y[i, 0])] = 1
                except IndexError:
                    pass
                # y_true[i, int(test_y[i, 0])] = 1
            # print('y_true', y_true)
            # print('y_pred',y_pred)

            try:
                auc = roc_auc_score(y_true, y_pred, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="ovo", labels=None)
            except ValueError:
                pass

            # auc = roc_auc_score(y_true, y_pred, average="macro", sample_weight=None,
            #       max_fpr=None, multi_class="ovo", labels=None)
            temp_aucs.append(auc)

            y_pred = clf.predict(test_vec)
            acc = accuracy_score(test_y, y_pred)
            temp_accs.append(acc)
            # print('y_pred',y_pred)
            mac = f1_score(test_y, y_pred,average='macro')
            temp_macro_f1.append(mac)
            mic = f1_score(test_y, y_pred,average='micro')
            temp_micro_f1.append(mic)

        average_acc = statistics.mean(temp_accs)
        average_accs.append(average_acc)

        average_auc = statistics.mean(temp_aucs)
        average_aucs.append(average_auc)

        average_macro_f1 = statistics.mean(temp_macro_f1)
        macro_f1.append(average_macro_f1)

        average_micro_f1 = statistics.mean(temp_micro_f1)
        micro_f1.append(average_micro_f1)

    return average_accs, average_aucs,macro_f1,micro_f1


def k_means(n_clu,emb,label):
    kmeans = KMeans(n_clusters=n_clu).fit(emb)
    label_pre = np.array(kmeans.labels_)
    label = label.reshape(-1)
    nmi = NMI(label_pre,label)
    return nmi


def NMI(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def Q_value(graph, cluster):
    q = 0
    num_edge = graph.number_of_edges() 
    num_node = graph.number_of_nodes()
    for i in range(num_node):
        for j in range(num_node):
            if cluster[i] == cluster[j]:
                q = q + graph.has_edge(i, j) - (graph.degree[i] * graph.degree[j]) / (2 * num_edge)
    q = q / (2 * num_edge)
    return q




