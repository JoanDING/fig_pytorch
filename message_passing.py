import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F

class BaseMean(nn.Module):
  def __init__(self, opt):
    super(BaseMean, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)

  def forward(self, embeddings):
    mu = embeddings.mean(dim=1,keepdim=True)
    # mu = mu.unsqueeze(dim=1)
    mu_exp = mu.expand_as(embeddings)
    adj_matrix = embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)
    return adj_matrix

class TransMean(nn.Module):
  def __init__(self, opt):
    super(TransMean,self).__init__()
    em_dim = opt.em_dim
    self.trans_i = nn.Linear(em_dim, em_dim)
    self.trans_m = nn.Linear(em_dim, em_dim)


  def forward(self, embeddings):
    mu = embeddings.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(embeddings)
    ctx_vec = embeddings * mu_exp

    h_i = self.trans_i(embeddings)
    trans_ctx = self.trans_m(ctx_vec)
    return h_i + trans_ctx



class Scorers(nn.Module):
  def  __init__(self, opt):
    super(Scorers, self).__init__()
    em_dim = opt.em_dim
    num_node = opt.num_node
    self.DEVICE = opt.DEVICE
    self.share_scorer = opt.share_scorer
    self.classme = opt.classme
    if self.share_scorer:
      self.scorers = nn.Linear(em_dim,1)
      print('{} scorers for embeddings scoring'.format(1))
    else:
      self.scorers = nn.ModuleList([nn.Linear(em_dim,1) for i in range(num_node)])
      print('{} scorers for embeddings scoring'.format(len(self.scorers)))
    if self.classme:
      self.cls_layer = nn.Linear(num_node, 1)


  def forward(self, embeddings, mask):
    bs, num_node, em_dim = embeddings.size()
    if self.share_scorer:
      prob = self.scorers(embeddings)
      prob = torch.squeeze(prob,dim=2)

    else:
      prob = torch.zeros([bs, num_node],device=self.DEVICE)
      for i in range(num_node):
        sc = self.scorers[i]
        em_i = embeddings[:,i,:]
        prob_i = sc(em_i)
        prob_i = torch.squeeze(prob_i,dim=1)
        prob[:, i] = prob_i
    prob = prob * mask # add for masking
    if self.classme:
      prob = self.cls_layer(prob)
      prob = torch.squeeze(prob,dim=1)
    else:
      prob = prob.sum(dim=1)/mask.sum(dim=1) # mean-->sum, for masking
    return prob


class Scorers2(nn.Module):
  def  __init__(self, opt):
    super(Scorers, self).__init__()
    em_dim = opt.em_dim
    num_node = opt.num_node
    self.DEVICE = opt.DEVICE
    self.share_scorer = opt.share_scorer
    self.classme = opt.classme
    if self.share_scorer:
      self.scorers = nn.Sequential(nn.Linear(em_dim,em_dim),nn.ReLU(),nn.Linear(em_dim,1))
      print('{} scorers for embeddings scoring'.format(1))
    else:
      self.scorers = nn.ModuleList([nn.Sequential(nn.Linear(em_dim,em_dim),nn.ReLU(),nn.Linear(em_dim,1)) for i in range(num_node)])
      print('{} scorers for embeddings scoring'.format(len(self.scorers)))
    if self.classme:
      self.cls_layer = nn.Linear(num_node, 1)

  def forward(self, embeddings):
    bs, num_node, em_dim = embeddings.size()
    if self.share_scorer:
      prob = self.scorers(embeddings)
      prob = torch.squeeze(prob,dim=2)
    else:
      prob = torch.zeros([bs, num_node],device=self.DEVICE)
      for i in range(num_node):
        sc = self.scorers[i]
        em_i = embeddings[:,i,:]
        prob_i = sc(em_i)
        prob_i = torch.squeeze(prob_i,dim=1)
        prob[:, i] = prob_i

    if self.classme:
      prob = self.cls_layer(prob)
      prob = torch.squeeze(prob,dim=1)
    else:
      prob = prob.mean(dim=1)
    return prob

