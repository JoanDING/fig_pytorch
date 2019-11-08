import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from base_att_module import *
import pdb

class att0_19_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_19_2layer, self).__init__()
        self.att_layer1 = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.att_layer2 = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer1(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)
        ui_att_adj1 = ui + att_value1
        em_out1 = self.activation_function(ui_att_adj1)
        em_out1 = self.drop(em_out1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer2(em_out1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)
        ui_att_adj = em_out1 + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out

class att0_18_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_18_2layer, self).__init__()
        self.att_layer = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(ui)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)
        ui_att_adj1 = ui + att_value1
        em_out1 = self.activation_function(ui_att_adj1)
        em_out1 = self.drop(em_out1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(em_out1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)
        ui_att_adj = em_out1 + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out


class att0_27_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_27_2layer, self).__init__()
        self.att_layer1 = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.att_layer2 = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer1(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer2(att_value1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)
        ui_att_adj = ui + att_value + att_value1
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out


class att0_26_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_26_2layer, self).__init__()
        self.att_layer = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(att_value1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)

        ui_att_adj = ui + att_value + att_value1
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out


class att0_28_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_28_2layer, self).__init__()
        self.att_layer1 = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.att_layer2 = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.activation_function(att_value1)
        att_value1 = self.drop(att_value1)
        att_value1 = self.mp_w1(att_value1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(att_value1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.activation_function(att_value)
        att_value = self.drop(att_value)
        att_value = self.mp_w2(att_value)

        ui_att_adj = ui + att_value + att_value1
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out

class att0_21_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_21_2layer, self).__init__()
        self.att_layer = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.activation_function(att_value1)
        att_value1 = self.drop(att_value1)
        att_value1 = self.mp_w1(att_value1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(att_value1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.activation_function(att_value)
        att_value = self.drop(att_value)
        att_value = self.mp_w2(att_value)

        ui_att_adj = ui + att_value + att_value1
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out

class att0_22_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_22_2layer, self).__init__()
        self.att_layer = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)
        att_value1 = att_value1 + ui

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(att_value1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)
        ui_att_adj = ui + att_value1 + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out


class att0_24_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_24_2layer, self).__init__()
        self.att_layer = Att_add_mp_norm(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(att_value1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)
        ui_att_adj = ui + att_value1 + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out

class att0_20_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_20_2layer, self).__init__()
        self.att_layer = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(att_value1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)
        ui_att_adj = ui + att_value1 + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out


class att0_23_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_23_2layer, self).__init__()
        self.att_layer = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)
        ui = u_em * att_value1
        em_out1 = self.activation_function(att_value1)
        em_out1 = self.drop(em_out1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(em_out1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)
        ui_att_adj = ui + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out


class att0_16_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_16_2layer, self).__init__()
        self.att_layer = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w1 = nn.Linear(opt.em_dim, opt.em_dim)
        self.mp_w2 = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w1(att_value1)
        ui_att_adj1 = ui + att_value1
        em_out1 = self.activation_function(ui_att_adj1)
        em_out1 = self.drop(em_out1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(em_out1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w2(att_value)
        ui_att_adj = em_out1 + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out

class att0_25_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_25_2layer, self).__init__()
        self.att_layer = Att_add_mp(opt.activation_fun, opt.dropout, opt.em_dim)
        self.mp_w = nn.Linear(opt.em_dim, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        u_em = embeddings[:,0,:]
        i_em = embeddings[:,1:,:]
        u_em = torch.unsqueeze(u_em,dim=-2)
        u_em = u_em.expand_as(i_em)
        ui = u_em * i_em

        # first layer ---------------------------------
        self.alphas1,value1 = self.att_layer(i_em)
        att_value1 = self.alphas1*value1
        att_value1 = att_value1.sum(dim=-2)
        att_value1 = self.mp_w(att_value1)
        ui_att_adj1 = ui + att_value1
        em_out1 = self.activation_function(ui_att_adj1)
        em_out1 = self.drop(em_out1)

        # secon layer ---------------------------------
        self.alphas,value = self.att_layer(em_out1)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w(att_value)
        ui_att_adj = em_out1 + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out


class att0_1_2layer(nn.Module):
    def __init__(self,opt):
        super(att0_1_2layer, self).__init__()
        self.att_layer = Att_cat(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        self.alphas2,value2 = self.att_layer(att_value)
        att_value2 = self.alphas2*value2
        att_value2 = att_value2.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out


class Scorers_w_id(nn.Module):
  def  __init__(self, opt):
    super(Scorers_w_id, self).__init__()
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
    self.activation = nn.Sigmoid()
    if self.classme:
      self.cls_layer = nn.Linear(num_node, 1)

  def forward(self, embeddings, mask):
    bs, num_node, em_dim = embeddings.size()

    prob = torch.zeros([bs, num_node],device=self.DEVICE)
    for i in range(num_node):
      sc = self.scorers[i]
      em_i = embeddings[:,i,:]
      prob_i = sc(em_i)
      prob_i = torch.squeeze(prob_i,dim=1)
      prob_i = self.activation(prob_i)
      prob[:, i] = prob_i
    prob = prob * mask # add for masking
    if self.classme:
      prob = self.cls_layer(prob)
      prob = torch.squeeze(prob,dim=1)
    else:
      prob = prob.sum(dim=1)/mask.sum(dim=1) # mean-->sum, for masking
    return prob


class Scorers_no_id(nn.Module):
  def  __init__(self, opt):
    super(Scorers_no_id, self).__init__()
    em_dim = opt.em_dim
    num_node = opt.num_node-1
    self.DEVICE = opt.DEVICE
    self.share_scorer = opt.share_scorer
    self.classme = opt.classme
    if self.share_scorer:
      self.scorers = nn.Linear(em_dim,1)
      print('{} scorers for embeddings scoring'.format(1))
    else:
      self.scorers = nn.ModuleList([nn.Linear(em_dim,1) for i in range(num_node)])
      print('{} scorers for embeddings scoring'.format(len(self.scorers)))
    self.activation = nn.Sigmoid()
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
        prob_i = self.activation(prob_i)
        prob[:, i] = prob_i
    prob = prob * mask # add for masking
    if self.classme:
      prob = self.cls_layer(prob)
      prob = torch.squeeze(prob,dim=1)
    else:
      prob = prob.sum(dim=1)/mask.sum(dim=1) # mean-->sum, for masking
    return prob

