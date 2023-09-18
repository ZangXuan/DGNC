import torch

def train(epoch, model, optimizer, train_adj, train_timestamp, onehot, num_nodes, out_dim, criterion):
    for epoch in range(epoch):
        acc, ap, f1, auc, m_loss = [], [], [], [], []

        train_snap = len(train_adj)
        z = [torch.FloatTensor(num_nodes, out_dim) for _ in range(train_snap)]

        for i in range(train_snap):
            if i == 0:
                z[i] = model(ini_feature=onehot, adj=train_adj[i], timestamp=train_timestamp[i])
            else:
                #z[i] = model(ini_feature=z[i-1], adj=train_adj[i], timestamp=train_timestamp[i])
                z[i] = model(ini_feature=onehot, adj=train_adj[i], timestamp=train_timestamp[i])

            # feed in the data and learn from error
            optimizer.zero_grad()
            model.train()
            loss = criterion(z, pos_label)
            loss.backward()
            optimizer.step()
            print('epoch', epoch, "loss=", "{:.5f}".format(loss.item()))



