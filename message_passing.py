import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F


class Attention(nn.Module):
  def __init__(self, opt):
    super(Attention, self).__init__()

  def forward(self, q,k,v):
    pass

class AttIJ(nn.Module):
  def __init__(self, opt):
    super(AttIJ, self).__init__()
    em_dim = opt.em_dim
    att_dim = opt.att_dim
    self.self_att = opt.self_att
    self.DEVICE = opt.DEVICE

    self.att_i = nn.Linear(em_dim, att_dim)
    self.att_j = nn.Linear(em_dim, att_dim)
    self.att_sum = nn.Linear(att_dim, 1)

    self.trans_i = nn.Linear(em_dim, em_dim)
    self.trans_att = nn.Linear(em_dim, em_dim)

  def forward(self, embeddings):
    '''
    update embeddings via message passing
    input: embeddings, batchsize*8*em_dim
    output: updated embeddings, batchsize*8*em_dim
    '''

    bs, item_dim, em_dim = embeddings.size()
    em_i = self.att_i(embeddings)
    em_j = self.att_j(embeddings)
    embeddings_exp = torch.unsqueeze(embeddings, dim=2)
    embeddings_exp = embeddings_exp.repeat([1,1,item_dim,1])


    em_i_exp = torch.unsqueeze(em_i, dim=1)
    em_i_exp = em_i_exp.repeat([1, item_dim, 1, 1])

    em_j_exp = torch.unsqueeze(em_j, dim=2)
    em_j_exp = em_j_exp.repeat([1, 1, item_dim, 1])
    att_z = self.att_sum(torch.tanh(em_i_exp + em_j_exp)) # with self attention
    att_z = torch.squeeze(att_z, dim=-1) # b*8*8

    h_i = self.trans_i(embeddings)

    if self.self_att:
      alphas = F.softmax(att_z, dim=2)
      ctx_vec = embeddings.expand_as(alphas)*alphas
      ctx_vec = torch.sum(ctx_vec, dim=2)
    else:
      att_z_slice = torch.zeros((bs, item_dim, item_dim-1),device=self.DEVICE) # bx8x7
      em_slice = torch.zeros((bs, item_dim, item_dim-1, em_dim), device=self.DEVICE) # values, expanded embeddings.
      for i in range(item_dim):
        tmp = torch.cat((att_z[:,i,0:i],att_z[:,i,i+1:]),dim=1) ##
        att_z_slice[:,i,:] = tmp

        tmp_em = torch.cat((embeddings_exp[:,i,0:i,:],embeddings_exp[:,i,i+1:,:]),dim=1)
        em_slice[:,i,:,:] = tmp_em

      # print('shape of em_slice: {}'.format(em_slice.size()))
      alphas = F.softmax(att_z_slice, dim=2)
      alphas = torch.unsqueeze(alphas,dim=-1).repeat([1,1,1,em_dim])
      # print('shape of alphas: {}'.format(alphas.size()))
      ctx_vec = alphas*em_slice
      ctx_vec = torch.sum(ctx_vec, dim=2)

    att_vec = self.trans_att(ctx_vec)
    return h_i + att_vec


import pdb
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

