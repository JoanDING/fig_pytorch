import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
import pdb

class ATT0(nn.Module):
  def __init__(self, opt):
    super(ATT0, self).__init__()
    self.em_dim = opt.em_dim
    self.DEVICE = opt.DEVICE
    self.drop = nn.Dropout(opt.dropout)

  def forward(self,embeddings):
    # input: old embeddings
    # output: updated embeddings
    bs, item_dim, em_dim = embeddings.size()

    # calculate the attentive mu
    em_i_exp = torch.unsqueeze(embeddings, dim=1)
    em_i_exp = em_i_exp.repeat([1, item_dim, 1, 1])
    em_j_exp = torch.unsqueeze(embeddings, dim=2)
    em_j_exp = em_j_exp.repeat([1, 1, item_dim, 1])
    em_ij = em_i_exp * em_j_exp

    ctx_vec = torch.mean(em_ij, dim=-2)
    ctx_vec = self.activation_function(ctx_vec)
    ctx_vec = self.drop(ctx_vec)
    return ctx_vec

class ATT1(nn.Module):
  def __init__(self, opt):
    super(ATT1, self).__init__()
    self.em_dim = opt.em_dim
    self.att_dim = self.em_dim*3
    self.self_att = opt.self_att
    self.DEVICE = opt.DEVICE
    self.activation_att = opt.activation_att
    self.drop = nn.Dropout(opt.dropout)
    self.drop_att = nn.Dropout(opt.dropout_att)
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()

    self.att_ij = nn.Linear(self.att_dim,self.em_dim)
    self.att_sum = nn.Linear(self.em_dim, 1)
    self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self,embeddings):
    # input: old embeddings
    # output: updated embeddings
    bs, item_dim, em_dim = embeddings.size()

    # calculate the attentive mu
    em_i_exp = torch.unsqueeze(embeddings, dim=1)
    em_i_exp = em_i_exp.repeat([1, item_dim, 1, 1])
    em_j_exp = torch.unsqueeze(embeddings, dim=2)
    em_j_exp = em_j_exp.repeat([1, 1, item_dim, 1])
    em_ij = em_i_exp * em_j_exp

    em_i_norm = self.layer_norm(em_i_exp)
    em_j_norm = self.layer_norm(em_j_exp)
    em_ij_norm = self.layer_norm(em_ij)
    em_i_j_ij = torch.cat((em_i_norm,em_j_norm,em_ij_norm),dim=-1)

    att_em = self.att_ij(em_i_j_ij)
    if self.activation_att == 'tanh':
        att_em = torch.tanh(att_em)
    att_em = self.drop_att(att_em)
    att_em = self.att_sum(att_em)
    att_em = torch.squeeze(att_em, dim=-1) # b*8*8

    if self.self_att:
      alphas = F.softmax(att_em, dim=-1)
      alphas = torch.unsqueeze(alphas,dim=-1).repeat([1,1,1,em_dim])
      ctx_vec = alphas*em_ij
      ctx_vec = self.activation_function(ctx_vec)
      ctx_vec = self.drop(ctx_vec)
      ctx_vec = torch.sum(ctx_vec, dim=2)

    else:
      att_z_slice = torch.zeros((bs, item_dim, item_dim-1),device=self.DEVICE) # bx8x7
      em_slice = torch.zeros((bs, item_dim, item_dim-1, em_dim), device=self.DEVICE) # values, expanded embeddings.
      for i in range(item_dim):
        tmp = torch.cat((att_em[:,i,0:i],att_em[:,i,i+1:]),dim=1) ##
        att_z_slice[:,i,:] = tmp
        tmp_em = torch.cat((em_ij[:,i,0:i,:],em_ij[:,i,i+1:,:]),dim=1)
        em_slice[:,i,:,:] = tmp_em

      alphas0 = F.softmax(att_z_slice, dim=-1)
      alphas = torch.unsqueeze(alphas0,dim=-1).repeat([1,1,1,em_dim])
      self.alphas = alphas
      ctx_vec = alphas*em_slice
      ctx_vec = torch.sum(ctx_vec, dim=-2)
      ctx_vec = self.activation_function(ctx_vec)
      ctx_vec = self.drop(ctx_vec)
    return ctx_vec


class ATT2(nn.Module):
  def __init__(self, opt):
    super(ATT2, self).__init__()
    self.em_dim = opt.em_dim
    self.att_dim = self.em_dim*3
    self.self_att = opt.self_att
    self.DEVICE = opt.DEVICE
    self.activation_att = opt.activation_att
    self.drop = nn.Dropout(opt.dropout)
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()

    self.att_ij = nn.Linear(self.att_dim,self.em_dim)
    self.att_sum = nn.Linear(self.em_dim, 1)
    self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self,embeddings):
    # input: old embeddings
    # output: updated embeddings
    bs, item_dim, em_dim = embeddings.size()

    # calculate the attentive mu
    em_i_exp = torch.unsqueeze(embeddings, dim=1)
    em_i_exp = em_i_exp.repeat([1, item_dim, 1, 1])
    em_j_exp = torch.unsqueeze(embeddings, dim=2)
    em_j_exp = em_j_exp.repeat([1, 1, item_dim, 1])
    em_ij = em_i_exp * em_j_exp

    em_i_norm = self.layer_norm(em_i_exp)
    em_j_norm = self.layer_norm(em_j_exp)
    em_ij_norm = self.layer_norm(em_ij)
    em_i_j_ij = torch.cat((em_i_norm,em_j_norm,em_ij_norm),dim=-1)

    att_em = self.att_ij(em_i_j_ij)
    if self.activation_att == 'tanh':
        att_em = torch.tanh(att_em)
    att_em = self.drop(att_em)
    att_em = self.att_sum(att_em)
    att_em = torch.squeeze(att_em, dim=-1) # b*8*8

    att_z_slice = torch.zeros((bs, item_dim, item_dim-1),device=self.DEVICE) # bx8x7
    em_slice = torch.zeros((bs, item_dim, item_dim-1, em_dim), device=self.DEVICE) # values, expanded embeddings.
    self_em = torch.zeros((bs,item_dim,em_dim),device=self.DEVICE)
    for i in range(item_dim):
        tmp = torch.cat((att_em[:,i,0:i],att_em[:,i,i+1:]),dim=1) ##
        att_z_slice[:,i,:] = tmp
        tmp_em = torch.cat((em_ij[:,i,0:i,:],em_ij[:,i,i+1:,:]),dim=1)
        self_em[:,i,:] = em_ij[:,i,i,:]
        em_slice[:,i,:,:] = tmp_em
        alphas0 = F.softmax(att_z_slice, dim=-1)
        alphas = torch.unsqueeze(alphas0,dim=-1).repeat([1,1,1,em_dim])
        ctx_vec = alphas*em_slice
        ctx_vec = torch.sum(ctx_vec, dim=-2)
        ctx_vec += self_em
        ctx_vec = self.activation_function(ctx_vec)
        ctx_vec = self.drop(ctx_vec)
    return ctx_vec


class AttIJ_plain(nn.Module):
  def __init__(self, opt):
    super(AttIJ_plain, self).__init__()
    em_dim = opt.em_dim
    att_dim = opt.att_dim
    self.self_att = opt.self_att
    self.DEVICE = opt.DEVICE

    self.att_ij = nn.Linear(em_dim, att_dim)
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

    em_i_exp = torch.unsqueeze(embeddings, dim=1)
    em_i_exp = em_i_exp.repeat([1, item_dim, 1, 1])
    em_j_exp = torch.unsqueeze(embeddings, dim=2)
    em_j_exp = em_j_exp.repeat([1, 1, item_dim, 1])

    att_z_0 = self.att_ij(em_i_exp + em_j_exp) # with self attention
    att_z_1 = em_i_exp * em_j_exp # with self attention
    att_z = torch.mean(torch.tanh(att_z_0),dim = -1) # with self attention
    att_z = torch.squeeze(att_z, dim=-1) # b*8*8

    #h_i = self.trans_i(embeddings)
    h_i = embeddings
    if self.self_att:
      alphas = F.softmax(att_z, dim=2)
      ctx_vec = embeddings.expand_as(alphas)*alphas
      ctx_vec = torch.sum(ctx_vec, dim=2)
    else:
      att_z_slice = torch.zeros((bs, item_dim, item_dim-1),device=self.DEVICE) # bx8x7
      em_slice = torch.zeros((bs, item_dim, item_dim-1, em_dim), device=self.DEVICE) # values, expanded embeddings.
      embeddings_exp = att_z_1
      for i in range(item_dim):
        tmp = torch.cat((att_z[:,i,0:i],att_z[:,i,i+1:]),dim=1) ##
        att_z_slice[:,i,:] = tmp
        tmp_em = torch.cat((embeddings_exp[:,i,0:i,:],embeddings_exp[:,i,i+1:,:]),dim=1)
        em_slice[:,i,:,:] = tmp_em

      # print('shape of em_slice: {}'.format(em_slice.size()))
      alphas0 = F.softmax(att_z_slice, dim=2)
      alphas = torch.unsqueeze(alphas0,dim=-1).repeat([1,1,1,em_dim])
      # crint('shape of alphas: {}'.format(alphas.size()))
      ctx_vec = alphas*em_slice
      ctx_vec = torch.sum(ctx_vec, dim=2)
    att_vec = self.trans_att(ctx_vec)
    return att_vec

class ATT3(nn.Module):
  def __init__(self, opt):
    super(ATT3, self).__init__()
    self.em_dim = opt.em_dim
    self.att_dim = self.em_dim*3
    self.self_att = opt.self_att
    self.DEVICE = opt.DEVICE
    self.activation_att = opt.activation_att
    self.drop = nn.Dropout(opt.dropout)
    self.drop_att = nn.Dropout(opt.dropout_att)
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()

    self.att_ij = nn.Linear(self.att_dim,self.em_dim)
    self.att_sum = nn.Linear(self.em_dim, 1)
    self.layer_norm = nn.LayerNorm(self.em_dim)
    self.self_trans = nn.Linear(self.em_dim,self.em_dim)

  def forward(self,embeddings):
    # input: old embeddings
    # output: updated embeddings
    bs, item_dim, em_dim = embeddings.size()

    # calculate the attentive mu
    em_i_exp = torch.unsqueeze(embeddings, dim=1)
    em_i_exp = em_i_exp.repeat([1, item_dim, 1, 1])
    em_j_exp = torch.unsqueeze(embeddings, dim=2)
    em_j_exp = em_j_exp.repeat([1, 1, item_dim, 1])
    em_ij = em_i_exp * em_j_exp

    em_i_norm = self.layer_norm(em_i_exp)
    em_j_norm = self.layer_norm(em_j_exp)
    em_ij_norm = self.layer_norm(em_ij)
    em_i_j_ij = torch.cat((em_i_norm,em_j_norm,em_ij_norm),dim=-1)

    att_em = self.att_ij(em_i_j_ij)
    if self.activation_att == 'tanh':
        att_em = torch.tanh(att_em)
    att_em = self.drop_att(att_em)
    att_em = self.att_sum(att_em)
    att_em = torch.squeeze(att_em, dim=-1) # b*8*8

    if self.self_att:
      alphas = F.softmax(att_em, dim=-1)
      alphas = torch.unsqueeze(alphas,dim=-1).repeat([1,1,1,em_dim])
      ctx_vec = alphas*em_ij
      ctx_vec = self.activation_function(ctx_vec)
      ctx_vec = self.drop(ctx_vec)
      ctx_vec = torch.sum(ctx_vec, dim=2)

    else:
      att_z_slice = torch.zeros((bs, item_dim, item_dim-1),device=self.DEVICE) # bx8x7
      em_slice = torch.zeros((bs, item_dim, item_dim-1, em_dim), device=self.DEVICE) # values, expanded embeddings.
      for i in range(item_dim):
        tmp = torch.cat((att_em[:,i,0:i],att_em[:,i,i+1:]),dim=1) ##
        att_z_slice[:,i,:] = tmp
        tmp_em = torch.cat((em_ij[:,i,0:i,:],em_ij[:,i,i+1:,:]),dim=1)
        em_slice[:,i,:,:] = tmp_em

      alphas0 = F.softmax(att_z_slice, dim=-1)
      alphas = torch.unsqueeze(alphas0,dim=-1).repeat([1,1,1,em_dim])
      self.alphas = alphas
      ctx_vec = alphas*em_slice
      ctx_vec = torch.sum(ctx_vec, dim=-2)
      ctx_vec = self.activation_function(ctx_vec)
      ctx_vec = self.drop(ctx_vec)
    self_trans = self.self_trans(embeddings)
    return ctx_vec + self_trans

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

    h_i = self.trans_i(embeddings) # b*8*64

    import pdb
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
      # att_z_slice: b*8*7
      # em_slice: b*8*7*64
      # print('shape of alphas: {}'.format(alphas.size()))
      ctx_vec = alphas*em_slice
      ctx_vec = torch.sum(ctx_vec, dim=2)

    att_vec = self.trans_att(ctx_vec)
    return h_i + att_vec

class Attention(nn.Module):
  def __init__(self, opt):
    super(Attention, self).__init__()
    em_dim = opt.em_dim
    att_dim = opt.att_dim
    self.self_att = opt.self_att
    self.DEVICE = opt.DEVICE

    self.trans_q = nn.Linear(em_dim,att_dim)
    self.trans_k = nn.Linear(em_dim,att_dim)
    self.trans_v = nn.Linear(em_dim,att_dim)

  def forward(self, q, k, v):
    pass

