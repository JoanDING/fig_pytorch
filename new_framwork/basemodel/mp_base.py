import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
import pdb

class BaseMean_w_id_6(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_w_id_6, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    mu = embeddings.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(embeddings)
    adj_matrix = embeddings * mu_exp
    adj = adj_matrix[:,1:,:]
    adj_matrix = self.activation_function(adj)
    out_embeddings = self.drop(adj_matrix)
    return out_embeddings


class BaseMean_10160_2(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_10160_2, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.w1 = nn.Linear(opt.em_dim,opt.em_dim)
    self.w2 = nn.Linear(opt.em_dim,opt.em_dim)
    self.drop = nn.Dropout(opt.dropout)


  def forward(self, embeddings):
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em,dim=-2)
    u_em = u_em.expand_as(i_em)
    mu = i_em.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(i_em)
    adj = i_em * mu_exp
    adj = self.w1(adj)
    i_em_1 = i_em + adj
    mu_1 = i_em_1.mean(dim=1,keepdim=True)
    mu_exp_1 = mu_1.expand_as(i_em_1)
    adj1 = i_em_1 * mu_exp_1
    adj1 = self.w2(adj)
    ui = u_em * (i_em_1 + adj1)

    ui_adj = self.activation_function(ui)
    out_embeddings = self.drop(ui_adj)
    return out_embeddings

class BaseMean_10160(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_10160, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.w1 = nn.Linear(opt.em_dim,opt.em_dim)
    self.drop = nn.Dropout(opt.dropout)

  def forward(self, embeddings):
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em,dim=-2)
    u_em = u_em.expand_as(i_em)
    mu = i_em.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(i_em)
    adj = i_em * mu_exp
    adj = self.w1(adj)
    ui = u_em * (i_em + adj)
    ui_adj = self.activation_function(ui)
    out_embeddings = self.drop(ui_adj)
    return out_embeddings

class BaseMean_add_mp1(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_add_mp1, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.w1 = nn.Linear(opt.em_dim,opt.em_dim)
    self.w2 = nn.Linear(opt.em_dim,opt.em_dim)
    self.drop = nn.Dropout(opt.dropout)

  def forward(self, embeddings):
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em,dim=-2)
    u_em = u_em.expand_as(i_em)
    ui = u_em * i_em
    mu = i_em.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(i_em)
    adj = i_em * mu_exp
    adj = self.w1(adj)
    ii = self.w2(i_em + adj)
    ui_adj = ui + ii
    ui_adj = self.activation_function(ui_adj)
    out_embeddings = self.drop(ui_adj)
    return out_embeddings

class BaseMean_add_mp(nn.Module):
  #output: bs*7*64
  def __init__(self, opt):
    super(BaseMean_add_mp, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.w = nn.Linear(opt.em_dim,opt.em_dim)
    self.drop = nn.Dropout(opt.dropout)

  def forward(self, embeddings):
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em,dim=-2)
    u_em = u_em.expand_as(i_em)
    ui = u_em * i_em
    mu = i_em.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(i_em)
    adj = i_em * mu_exp
    adj = self.w(adj)
    ui_adj = ui + adj
    ui_adj = self.activation_function(ui_adj)
    out_embeddings = self.drop(ui_adj)
    return out_embeddings

class BaseMean_no_mp(nn.Module):
  def __init__(self, opt):
    super(BaseMean_no_mp, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    u_em = embeddings[:,0,:]
    i_em = embeddings[:,1:,:]
    u_em = torch.unsqueeze(u_em,dim=-2)
    u_em = u_em.expand_as(i_em)
    adj = u_em * i_em
    adj_matrix = self.activation_function(adj)
    out_embeddings = self.drop(adj_matrix)
    return out_embeddings

class BaseMean_w_id_5(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_5, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim
    self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = self.layer_norm(ui_embeddings)
    ua_out = self.layer_norm(adj_matrix[:,1:,:])
    ua_out = adj_matrix[:,1:,:]
    ui_out = ui_out.expand_as(ua_out)
    ua_out = torch.cat((ui_out, ua_out),dim = -1)
    out_embeddings = torch.cat((torch.cat((ui_embeddings,ui_embeddings),dim=-1),ua_out),dim=-2)
    return out_embeddings

class BaseMean_w_id_4(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_4, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim
    self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = self.layer_norm(ui_embeddings)
    ua_out = self.layer_norm(adj_matrix[:,1:,:])
    ua_out = adj_matrix[:,1:,:]
    ua_out = ui_out + ua_out
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings


class BaseMean_w_id_3(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_3, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim
    self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings_0 = torch.unsqueeze(u_embeddings, dim = -2)
    u_embeddings = u_embeddings_0.expand_as(a_embeddings)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = u_embeddings*a_embeddings
    mu = ua_embeddings.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings_0*i_embeddings
    ui_out = self.layer_norm(ui_embeddings)
    ua_out = adj_matrix
    ua_out = self.layer_norm(ua_out)
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings


class BaseMean_w_id_2(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_2, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings_0 = torch.unsqueeze(u_embeddings, dim = -2)
    u_embeddings = u_embeddings_0.expand_as(a_embeddings)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = u_embeddings*a_embeddings
    mu = ua_embeddings.mean(dim=1,keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings_0*i_embeddings
    ui_out = ui_embeddings
    ua_out = adj_matrix
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

# type 1------------------------------------------------------------------
# ------------------------------------------------------------------------
class BaseMean_w_id_1_6(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_1_6, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.layer_norm = nn.LayerNorm(opt.em_dim)
    self.trans = nn.Linear(opt.em_dim, opt.em_dim)
    self.trans_ui = nn.Linear(opt.em_dim, opt.em_dim)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim = -2, keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = ui_embeddings
    ui_out = self.trans_ui(ui_out)
    ua_out = adj_matrix[:,1:,:]
    ua_out = self.trans(ua_out)
    ua_out = ua_out + a_embeddings
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

class BaseMean_w_id_1_5(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_1_5, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.layer_norm = nn.LayerNorm(opt.em_dim)
    self.trans = nn.Linear(opt.em_dim, opt.em_dim)
    self.trans_ui = nn.Linear(opt.em_dim, 2*opt.em_dim)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim = -2, keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = self.trans_ui(ui_embeddings)
    ua_out = adj_matrix[:,1:,:]
    ua_out = self.trans(ua_out)
    ua_out = torch.cat((ua_out,a_embeddings),dim = -1)
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

class BaseMean_w_id_1_4(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_1_4, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.layer_norm = nn.LayerNorm(opt.em_dim)
    self.trans = nn.Linear(opt.em_dim, opt.em_dim)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim = -2, keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = torch.cat((ui_embeddings,ui_embeddings),dim = -1)
    ua_out = adj_matrix[:,1:,:]
    ua_out = self.trans(ua_out)
    ua_out = torch.cat((ua_out,a_embeddings),dim = -1)
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings

class BaseMean_w_id_1_3(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_1_3, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.layer_norm = nn.LayerNorm(opt.em_dim)
    self.trans = nn.Linear(opt.em_dim, opt.em_dim)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim = -2, keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = ui_embeddings
    ua_out = adj_matrix[:,1:,:]
    ua_out = self.trans(ua_out)
    ua_out = ua_out + a_embeddings
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings


class BaseMean_w_id_1_2(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_1_2, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.layer_norm = nn.LayerNorm(opt.em_dim)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim = -2, keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = ui_embeddings
    ua_out = adj_matrix[:,1:,:]
    ui_out = self.layer_norm(ui_out)
    ua_out = self.layer_norm(ua_out)
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings


class BaseMean_w_id_1(nn.Module):
  def __init__(self, opt):
    super(BaseMean_w_id_1, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim = -2, keepdim=True)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)

    ui_embeddings = u_embeddings*i_embeddings
    ui_out = ui_embeddings
    ua_out = adj_matrix[:,1:,:]
    out_embeddings = torch.cat((ui_out, ua_out),dim = -2)
    return out_embeddings


class BaseMean_only_id(nn.Module):
  def __init__(self, opt):
    super(BaseMean_only_id, self).__init__()
    #self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self, embeddings):
    u_embeddings = embeddings[:,0]
    i_embeddings = embeddings[:,1]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    i_embeddings = torch.unsqueeze(i_embeddings, dim = -2)

    ui_embeddings = u_embeddings*i_embeddings
    #ui_out = ui_embeddings.sum(dim=-1)
    #ui_out = torch.squeeze(ui_out, dim=-1)
    ui_out = ui_embeddings
    return ui_out

class BaseMean_no_id(nn.Module):
  def __init__(self, opt):
    super(BaseMean_no_id, self).__init__()
    if opt.activation_fun == 'leaky_relu':
        self.activation_function = nn.LeakyReLU()
    self.drop = nn.Dropout(opt.dropout)
    self.em_dim = opt.em_dim
    self.layer_norm = nn.LayerNorm(self.em_dim)

  def forward(self, embeddings):
    #-----------------------------------------------------------------------------------------
    # input: all embeddings, including user id, item id and attributes, input size: 8*emb_size
    # output: updated attributes embeddings output size: 6*emb_size
    #-----------------------------------------------------------------------------------------
    u_embeddings = embeddings[:,0]
    a_embeddings = embeddings[:,2:]
    u_embeddings = torch.unsqueeze(u_embeddings, dim = -2)
    ua_embeddings = torch.cat((u_embeddings,a_embeddings),dim = -2)
    mu = ua_embeddings.mean(dim=1,keepdim=True)
    # mu = mu.unsqueeze(dim=1)
    mu_exp = mu.expand_as(ua_embeddings)
    adj_matrix = ua_embeddings * mu_exp
    adj_matrix = self.activation_function(adj_matrix)
    adj_matrix = self.drop(adj_matrix)
    out_embeddings = adj_matrix[:,1:]
    return out_embeddings

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

