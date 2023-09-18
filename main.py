import torch
import numpy as np
import warnings
import random
from model import Model
from args import set_params
from data_load import load_profile,neg_sample,load_labels
from task import link_prediction,node_classification,k_means

warnings.filterwarnings('ignore')
args = set_params()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

own_str = args.dataset

## random seed ##
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

adj, timestamp, onehot,graphs = load_profile(args.dataset,args.t_span)
train_adj = adj[0:args.train_span]
train_timestamp = timestamp[0:args.train_span]
neg = neg_sample(train_adj,args.num_nodes)


model = Model(args.num_nodes, args.out_dim, args.K, args.tau, args.lam, args.feat_drop, args.attn_drop)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


if __name__ == '__main__':
    print(args)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        adj = [adj.cuda() for adj in adj]
        timestamp = [time.cuda() for time in timestamp]
        onehot = onehot.cuda()
        train_adj = [i.cuda() for i in train_adj]
        train_timestamp = [i.cuda() for i in train_timestamp]
        neg = [i.cuda() for i in neg]

    best_epoch_val = 0
    patient = 0

    for epoch in range(args.epoch):
        print('====epoch {}'.format(epoch))

        optimizer.zero_grad()
        emb_z, emb_h, a_z, a_h = model(train_adj, onehot, train_timestamp, args.t_span)
        loss = model.loss(emb_z, emb_h, train_adj, neg)
        loss.backward()
        optimizer.step()

        model.eval()
        emb_next,attn_next = model.get_next(adj[args.train_span], emb_z[-1], emb_h[-1], timestamp[args.train_span], a_z,
                                  args.train_span,args.t_span)

        epoch_auc_val,epoch_auc_test,epoch_ap_val,epoch_ap_test = link_prediction(emb_next, adj[args.train_span])
        if epoch_auc_val > best_epoch_val:
            best_epoch_val = epoch_auc_val
            torch.save(model.state_dict(), './model_checkpoints/model_' + args.dataset + '.pkl')
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print('patient:',patient)
        print("Epoch {:<3}: Loss = {:.3f}, Val AUC {:.4f} Test AUC {:.4f}, Val AP {:.4f} Test AP {:.4f}".format(epoch,
                                                                                   loss.data,
                                                                                   epoch_auc_val,epoch_auc_test,epoch_ap_val,epoch_ap_test))

    model.load_state_dict(torch.load('./model_checkpoints/model_'+args.dataset+'.pkl'))
    model.eval()

    emb_z, emb_h,a_z, a_h = model(train_adj,onehot,train_timestamp,args.t_span)
    emb_next,attn_next = model.get_next(adj[args.train_span], emb_z[-1], emb_h[-1], timestamp[args.train_span], a_z,
                              args.train_span, args.t_span)
    val_roc_score,test_roc_score,val_AP_score,test_AP_score = link_prediction(emb_next, adj[args.train_span])

    print("Best Test AUC = {:.4f}, Best Test AP = {:.4f}".format(test_roc_score,test_AP_score))

    if args.node_clf:
        labels = load_labels(args.dataset, args.num_nodes)
        nmi = k_means(args.K, np.array(emb_next.cpu().detach()), labels)
        print("nmi = {:.4f}".format(nmi))
        acc,auc,mac_f1, mic_f1 = node_classification(emb_next, labels, args.num_nodes)
        print("ACC(0.3) = {:.4f}, ACC(0.5) = {:.4f}, ACC(0.7) = {:.4f}".format(acc[0],acc[1],acc[2]))
        print("AUC(0.3) = {:.4f}, AUC(0.5) = {:.4f}, AUC(0.7) = {:.4f}".format(auc[0],auc[1],auc[2]))
        print("mac(0.3) = {:.4f}, mac(0.5) = {:.4f}, mac(0.7) = {:.4f}".format(mac_f1[0],mac_f1[1],mac_f1[2]))
        print("mic(0.3) = {:.4f}, mic(0.5) = {:.4f}, mic(0.7) = {:.4f}".format(mic_f1[0],mic_f1[1],mic_f1[2]))


