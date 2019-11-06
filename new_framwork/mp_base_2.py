import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
import pdb

# message passing---------------------------------------------------------------------
#---------------------------------------------------------------------------------------

class base_mp_6(nn.Module):
    def __init__(self, opt):
        super(base_mp_6, self).__init__()
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.em_dim = opt.em_dim

    def forward(self,embeddings):
        mu = embeddings.mean(dim=1, keepdim=True)
        mu_exp = mu.expand_as(embeddings)
        adj_matrix = embeddings * mu_exp
        adj_matrix = self.activation_function(adj_matrix)
        adj_matrix = self.drop(adj_matrix)
        mu2 = adj_matrix.mean(dim=1, keepdim=True)
        mu_exp2 = mu2.expand_as(adj_matrix)
        adj_matrix2 = adj_matrix * mu_exp2
        adj_matrix2 = self.activation_function(adj_matrix2)
        adj_matrix2 = self.drop(adj_matrix2)
        out_embeddings = adj_matrix[:,1:,:]
        return out_embeddings


class BaseMean_add_mp_3(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_add_mp_3, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.w0 = nn.Linear(opt.em_dim,opt.em_dim)
    self.w1 = nn.Linear(opt.em_dim,opt.em_dim)
    self.drop = nn.Dropout(opt.dropout)

  def forward(self, embeddings):
    # first layer ---------------
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em,dim=-2)
    u_em = u_em.expand_as(i_em)
    ui0 = u_em * i_em
    mu0 = i_em.mean(dim=1,keepdim=True)
    mu0_exp = mu0.expand_as(i_em)
    adj0 = i_em * mu0_exp
    adj0 = self.w0(adj0)
    ui_adj0 = ui0 + adj0

    # second layer ---------------
    mu1 = ui_adj0.mean(dim=1,keepdim=True)
    mu1_exp = mu1.expand_as(ui_adj0)
    adj1 = ui_adj0 * mu1_exp
    adj1 = self.w1(adj1)
    ui_adj1 = ui_adj0 + adj1
    ui_adj1 = self.activation_function(ui_adj1)
    ui_adj1 = self.drop(ui_adj1)

    out_embeddings = ui_adj1
    return out_embeddings

class BaseMean_add_mp_2(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_add_mp_2, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.w0 = nn.Linear(opt.em_dim,opt.em_dim)
    self.w1 = nn.Linear(opt.em_dim,opt.em_dim)
    self.drop = nn.Dropout(opt.dropout)

  def forward(self, embeddings):
    # first layer ---------------
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em,dim=-2)
    u_em = u_em.expand_as(i_em)
    ui0 = u_em * i_em
    mu0 = i_em.mean(dim=1,keepdim=True)
    mu0_exp = mu0.expand_as(i_em)
    adj0 = i_em * mu0_exp
    adj0 = self.w0(adj0)
    ui_adj0 = ui0 + adj0
    ui_adj0 = self.activation_function(ui_adj0)
    ui_adj0 = self.drop(ui_adj0)

    # second layer ---------------
    mu1 = ui_adj0.mean(dim=1,keepdim=True)
    mu1_exp = mu1.expand_as(ui_adj0)
    adj1 = ui_adj0 * mu1_exp
    adj1 = self.w1(adj1)
    ui_adj1 = ui_adj0 + adj1
    ui_adj1 = self.activation_function(ui_adj1)
    ui_adj1 = self.drop(ui_adj1)

    out_embeddings = ui_adj1
    return out_embeddings

class BaseMean_add_mp_1(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_add_mp_1, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.w0 = nn.Linear(opt.em_dim,opt.em_dim)
    self.w1 = nn.Linear(opt.em_dim,opt.em_dim)
    self.drop = nn.Dropout(opt.dropout)

  def forward(self, embeddings):
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em,dim=-2)
    u_em = u_em.expand_as(i_em)
    ui0 = u_em * i_em
    # first layer ---------------
    mu0 = i_em.mean(dim=1,keepdim=True)
    mu0_exp = mu0.expand_as(i_em)
    adj0 = i_em * mu0_exp
    adj0 = self.w0(adj0)
    # second layer ---------------
    mu1 = adj0.mean(dim=1,keepdim=True)
    mu1_exp = mu1.expand_as(adj0)
    adj1 = adj0 * mu1_exp
    adj1 = self.w1(adj1)

    ui_adj = ui0 + adj0 + adj1
    ui_adj = self.activation_function(ui_adj)
    ui_adj = self.drop(ui_adj)

    out_embeddings = ui_adj
    return out_embeddings

class base_mp(nn.Module):
    def __init__(self,activation,dropout):
        super(base_mp,self).__init__()
        if activation == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,ems):
        mu = ems.mean(dim=1,keepdim=True)
        mu_exp = mu.expand_as(ems)
        adj_matrix = ems * mu_exp
        adj_matrix = self.activation_function(adj_matrix)
        adj_matrix = self.drop(adj_matrix)
        return adj_matrix

class base_mp_norm(nn.Module):
    def __init__(self,activation,dropout,em_dim):
        super(base_mp_norm,self).__init__()
        if activation == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(em_dim)

    def forward(self,ems):
        mu = ems.mean(dim=1,keepdim=True)
        mu_exp = mu.expand_as(ems)
        adj_matrix = ems * mu_exp
        adj_matrix = self.activation_function(adj_matrix)
        adj_matrix = self.drop(adj_matrix)
        adj_matrix = self.layer_norm(adj_matrix)
        return adj_matrix

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

class BaseMean_w_id_2_0(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_0, self).__init__()
    self.base_mp = base_mp(opt.activation_fun,opt.dropout)

  def forward(self,embeddings):
    adj_matrix = self.base_mp(embeddings)
    adj_matrix_2 = self.base_mp(adj_matrix)
    out_embeddings = adj_matrix_2[:,1:,:]
    return out_embeddings


class BaseMean_w_id_2_0_1(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_0_1, self).__init__()
    self.base_mp_norm = base_mp_norm(opt.activation_fun,opt.dropout,opt.em_dim)

  def forward(self,embeddings):
    adj_matrix = self.base_mp_norm(embeddings)
    adj_matrix_2 = self.base_mp_norm(adj_matrix)
    out_embeddings = adj_matrix_2[:,1:,:]
    return out_embeddings

class BaseMean_w_id_2_0_2(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_0_2, self).__init__()
    self.base_mp = base_mp(opt.activation_fun,opt.dropout)
    self.base_mp_norm = base_mp_norm(opt.activation_fun,opt.dropout,opt.em_dim)

  def forward(self,embeddings):
    adj_matrix = self.base_mp_norm(embeddings)
    adj_matrix_2 = self.base_mp(adj_matrix)
    out_embeddings = adj_matrix_2[:,1:,:]
    return out_embeddings

class BaseMean_w_id_2_0_3(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_0_3, self).__init__()
    self.base_mp = base_mp(opt.activation_fun,opt.dropout)
    self.base_mp_norm = base_mp_norm(opt.activation_fun,opt.dropout,opt.em_dim)

  def forward(self,embeddings):
    adj_matrix = self.base_mp(embeddings)
    adj_matrix_2 = self.base_mp_norm(adj_matrix)
    out_embeddings = adj_matrix_2[:,1:,:]
    return out_embeddings

class BaseMean_w_id_2_0_4(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_0_4, self).__init__()
    self.base_mp = base_mp(opt.activation_fun,opt.dropout)

  def forward(self,embeddings):
    adj_matrix = self.base_mp(embeddings)
    adj_matrix_2 = self.base_mp(adj_matrix)
    out_embeddings = torch.cat((adj_matrix[:,1:,:],adj_matrix_2[:,1:,:]),dim=-1)
    return out_embeddings

class BaseMean_w_id_2_0_5(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_0_5, self).__init__()
    self.base_mp_norm = base_mp_norm(opt.activation_fun,opt.dropout,opt.em_dim)

  def forward(self,embeddings):
    adj_matrix = self.base_mp_norm(embeddings)
    adj_matrix_2 = self.base_mp_norm(adj_matrix)
    out_embeddings = torch.cat((adj_matrix[:,1:,:],adj_matrix_2[:,1:,:]),dim=-1)
    return out_embeddings


class BaseMean_w_id_2_0_6(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_0_6, self).__init__()
    self.base_mp_norm = base_mp_norm(opt.activation_fun,opt.dropout,opt.em_dim)
    self.base_mp = base_mp(opt.activation_fun,opt.dropout)

  def forward(self,embeddings):
    adj_matrix = self.base_mp(embeddings)
    adj_matrix_2 = self.base_mp_norm(adj_matrix)
    out_embeddings = torch.cat((adj_matrix[:,1:,:],adj_matrix_2[:,1:,:]),dim=-1)
    return out_embeddings

class BaseMean_w_id_2_6(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_6, self).__init__()
    self.base_mp = base_mp(opt.activation_fun,opt.dropout)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings_0 = torch.unsqueeze(u_embeddings, dim = -2)
    u_embeddings = u_embeddings_0.expand_as(a_embeddings)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)
    ua_embeddings = u_embeddings*a_embeddings

    # mp---------------------------------------------------------------------------------
    adj_matrix = self.base_mp(ua_embeddings)
    adj_matrix_2 = self.base_mp(adj_matrix)
    # -----------------------------------------------------------------------------------

    ui_embeddings = u_embeddings_0*i_embeddings

    ui_out = ui_embeddings*ui_embeddings
    ua_out = adj_matrix_2
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

class BaseMean_w_id_2_5(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_5, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim
    self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2) #bs*1*64
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2) #bs*att_num*64
    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)

    # first layer mp---------------------------------------------------------------
    mu = ua_embeddings.mean(dim=1,keepdim=True) #bs*64
    mu_exp = mu.expand_as(ua_embeddings) #bs*(att_num+1)*64
    adj_matrix = ua_embeddings * mu_exp #bs*(att_num+1)*64

    # second layer mp---------------------------------------------------------------
    adj_mu = adj_matrix.mean(dim=1,keepdim=True) #bs*64
    adj_mu_exp = adj_mu.expand_as(ua_embeddings)
    adj_matrix_2 = adj_matrix*adj_mu_exp
    adj_matrix_2 = self.activation_function(adj_matrix_2)
    adj_matrix_2 = self.drop(adj_matrix_2)
    adj_matrix_2 = self.layer_norm(adj_matrix_2)

    ui_embeddings = u_embeddings*i_embeddings
    ui_embeddings = self.layer_norm(ui_embeddings)
    ui_out = torch.cat((ui_embeddings,ui_embeddings),dim=-1)
    ua_out = torch.cat((adj_matrix[:,1:,:],adj_matrix_2[:,1:,:]),dim=-1)
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

class BaseMean_w_id_2_4(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_4, self).__init__()
    self.base_mp = base_mp(opt.activation_fun,opt.dropout)
    self.trans_ui = nn.Linear(opt.em_dim,opt.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2) #bs*1*64
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2) #bs*att_num*64
    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)

    # mp---------------------------------------------------------------
    adj_matrix = self.base_mp(ua_embeddings)
    adj_matrix_2 = self.base_mp(adj_matrix)
    # -----------------------------------------------------------------

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = self.trans_ui(ui_embeddings)
    ua_out = adj_matrix_2[:,1:,:]
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

class BaseMean_w_id_2_3(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_3, self).__init__()
    self.base_mp_norm = base_mp_norm(opt.activation_fun,opt.dropout,opt.em_dim)
    self.layer_norm = nn.LayerNorm(opt.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2) #bs*1*64
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2) #bs*att_num*64
    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)

    # mp---------------------------------------------------------------
    adj_matrix = self.base_mp_norm(ua_embeddings)
    adj_matrix_2 = self.base_mp_norm(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = self.layer_norm(ui_embeddings)
    ua_out = adj_matrix[:,1:,:] + adj_matrix_2[:,1:,:]
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

class BaseMean_w_id_2_2(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_2, self).__init__()
    self.base_mp = base_mp(opt.activation_fun,opt.dropout)
    self.base_mp_norm = base_mp_norm(opt.activation_fun,opt.dropout,opt.em_dim)
    self.layer_norm = nn.LayerNorm(opt.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2) #bs*1*64
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2) #bs*att_num*64
    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)

    # mp---------------------------------------------------------------
    adj_matrix = self.base_mp(ua_embeddings)
    adj_matrix_2 = self.base_mp_norm(adj_matrix)
    # -----------------------------------------------------------------

    ui_embeddings = u_embeddings*i_embeddings
    ui_embeddings = self.layer_norm(ui_embeddings)
    ui_out = torch.cat((ui_embeddings,ui_embeddings),dim=-1)
    ua_out = torch.cat((adj_matrix[:,1:,:],adj_matrix_2[:,1:,:]),dim=-1)
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings


class BaseMean_w_id_2_1(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2_1, self).__init__()
    self.base_mp_norm = base_mp_norm(opt.activation_fun,opt.dropout,opt.em_dim)
    self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2) #bs*1*64
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2) #bs*att_num*64
    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)

    # mp---------------------------------------------------------------
    adj_matrix = self.base_mp(ua_embeddings)
    adj_matrix_2 = self.base_mp(adj_matrix)
    # -----------------------------------------------------------------

    ui_embeddings = u_embeddings*i_embeddings
    ui_embeddings = self.layer_norm(ui_embeddings)
    ui_out = torch.cat((ui_embeddings,ui_embeddings),dim=-1)
    ua_out = torch.cat((adj_matrix[:,1:,:],adj_matrix_2[:,1:,:]),dim=-1)
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

#-----------------------------------------------------------------------------------
#scores-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

class Scorers_only_id(nn.Module):
    def __init__(self,opt):
        super(Scorers_only_id,self).__init__()
        em_dim = opt.em_dim
        self.scorer = nn.Linear(em_dim,1)
        self.activation = nn.Sigmoid()

    def forward(self,embedding):
        prob = self.scorer(embedding)
        prob = self.activation(prob)
        return prob

class Scorers_w_id_sep(nn.Module):
  def  __init__(self, opt):
    super(Scorers_w_id_sep, self).__init__()
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
    prob_id = prob[:,0]
    prob_a = prob[:,1:].sum(dim=1)/mask[:,1:].sum(dim=1) # mean-->sum, for masking
    prob = (0.2*prob_id+0.8*prob_a)/2
    return prob


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

