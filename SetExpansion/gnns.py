from turtle import distance
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GATv2Conv, GCNConv
import torch.nn.functional as F

class GnnBase(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0, distance = 'correlation', sparse='knn',knn=80, threshold = 0.5):
        super(GnnBase, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.distance = distance
        self.sparse = sparse
        self.threshold = threshold
        self.knn = knn
        self.dropout = dropout
        self.model = self.get_model()
    
    def get_model(self):
        pass

    def compute_gso(self, x):
    # Compute GRM
        if self.distance == 'correlation':
            dense_GSO = torch.corrcoef(x)
        elif self.distance == 'l2':
            l2 = torch.cdist(x[None,:,:], x[None,:,:]).squeeze()
            dense_GSO = 1/(l2+1e-10)
        else:
            raise NotImplementedError
        if self.sparse=='knn':
            GSO = torch.zeros(dense_GSO.shape, device = dense_GSO.device)
            for i in range(GSO.shape[0]):
                closest = torch.argsort(dense_GSO[i])[1:1+self.knn]
                GSO[i][closest] = dense_GSO[i][closest]
            return GSO
        elif self.sparse=='None':
            return dense_GSO
        elif self.sparse=='threshold':
            return dense_GSO*(dense_GSO>self.threshold)
        else:    
            raise NotImplementedError 
    def batch_to_graph(self, x, y=None):
        GSO = self.compute_gso(x)
        edge_index, edge_attr = dense_to_sparse(GSO)
        return Data(x=x, edge_index=edge_index,edge_attr=edge_attr, y=y)

    def forward(self, x):
        return self.model(self.batch_to_graph(x))

class two_layer_GAT(torch.nn.Module):
    def __init__(self,embedding_dim, hidden_dim, dropout=0, batch_norm=False):
        super(two_layer_GAT, self).__init__()
        self.GAT1 = GATv2Conv(embedding_dim,
                            hidden_dim,
                            dropout = dropout)
        self.GAT2 = GATv2Conv(hidden_dim,
                                1,
                                dropout = dropout)
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(embedding_dim)
            self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.GAT1(x, edge_index)
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = self.GAT2(x, edge_index)
        return x

class one_layer_gcn(torch.nn.Module):
    def __init__(self,embedding_dim, dropout=0, batch_norm=False):
        super(one_layer_gcn, self).__init__()
        self.GCN = GCNConv(embedding_dim,1,)
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(embedding_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = self.GCN(x, edge_index)
        x = F.relu(x)
        return x

class GAtt(GnnBase):
    def __init__(self, embedding_dim, hidden_dim, dropout=0, distance = 'correlation', sparse='knn', threshold = 0.5):
        super(GAtt, self).__init__(embedding_dim, hidden_dim,dropout=dropout, distance=distance, sparse=sparse, threshold=threshold)
    
    def get_model(self):
        return two_layer_GAT(self.embedding_dim, self.hidden_dim, self.dropout)

class GCN(GnnBase):
    def __init__(self, embedding_dim, dropout=0, distance = 'l2', sparse='threshold', threshold = 0.1):
        super(GCN, self).__init__(embedding_dim, 0,dropout=dropout, distance=distance, sparse=sparse, threshold=threshold)
    
    def get_model(self):
        return one_layer_gcn(self.embedding_dim, self.dropout)      