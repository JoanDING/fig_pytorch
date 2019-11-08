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


class mul_attention(nn.Module):
    def __init__(self, opt):
        super(mul_attention, self).__init__()
        em_dim = opt.em_dim
        num_node = opt.num_node-1
        self.DEVICE = opt.DEVICE
        self.attention_act = opt.attention_act
        #self.att_W_q = nn.Linear(em_dim,em_dim)
        #self.att_W_k = nn.Linear(em_dim,em_dim)
        self.att_W_v = nn.Linear(em_dim,1)
        self.layer_norm = nn.LayerNorm(em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        query = embeddings[:,0,:]
        query = torch.unsqueeze(query, dim = -2)
        key = embeddings[:,1:,:]
        value = key
        q = query.expand_as(key)
        #q_pro = self.att_W_q(q)
        #k_pro = self.att_W_k(key)
        q_pro = q
        k_pro = key
        qk = q_pro * k_pro
        if self.attention_act == 'tanh':
            qk = torch.tanh(qk)
        qk = self.att_W_v(qk)
        qk = torch.squeeze(qk, dim=-1)
        alphas = F.softmax(qk, dim=-1)
        alphas = torch.unsqueeze(alphas,dim=-1)
        self.alphas = alphas.expand_as(value)
        em_a = self.alphas*value
        em_out = torch.cat((query,em_a),dim = -2)
        return em_out


class GAT_attention_multi(nn.Module):
    # conduct graph attention network(GAT), multi head
    def __init__(self,opt):
        super(GAT_attention_multi, self).__init__()
        em_dim = opt.em_dim
        num_node = opt.num_node-1
        self.DEVICE = opt.DEVICE
        self.att_W1 = nn.Linear(em_dim,em_dim)
        self.att_W2 = nn.Linear(em_dim,em_dim)
        self.att_a1 = nn.Linear(3*em_dim,1)
        self.att_a2 = nn.Linear(3*em_dim,1)
        self.layer_norm = nn.LayerNorm(em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        # normalize 1
        embeddings = self.layer_norm(embeddings)
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iid_0 = embeddings[:,1,:] #256,64
        iid = torch.unsqueeze(iid_0, dim = -2)
        iid = torch.unsqueeze(iid, dim = -2) #256,1,1,64
        iatt = embeddings[:,2:,:] #256,6,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,6,64
        # Calculating attention parameter------------------------------
        query = torch.unsqueeze(ua,dim=-2)
        key = torch.unsqueeze(ua,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        value = query * key
        value = self.layer_norm(value)
        iid = iid.expand_as(key)
        # key: 256,6,6,64; query:256,6,6,64; iid: 256,6,6,64

        # first head-------------------------
        self.wq = self.att_W1(query)
        self.wk = self.att_W1(key)
        self.wiid = self.att_W1(iid)
        qk1 = torch.cat((self.wq,self.wk,self.wiid),dim=-1) # 256,6,6,192
        qk1 = self.att_a1(qk1) #256,6,6,1
        qk1 = torch.squeeze(qk1,dim=-1) #256,6,6
        qk1 = self.activation_function(qk1)
        self.qk = self.drop(qk1)
        self.alphas = F.softmax(self.qk, dim=-1) #256,6,6
        alphas1 = torch.unsqueeze(self.alphas,dim=-1)
        # normalize 2, to make sure att_value and ui are not in different shuliangji
        alphas1 = alphas1.expand_as(value) #256,6,6,64
        att_value1 = alphas1*value
        att_value1 = att_value1.sum(dim=-2)

        # second head-------------------------
        self.wq2 = self.att_W2(query)
        self.wk2 = self.att_W2(key)
        self.wiid2 = self.att_W2(iid)
        qk2 = torch.cat((self.wq2,self.wk2,self.wiid2),dim=-1) # 256,6,6,192
        qk2 = self.att_a2(qk2) #256,6,6,1
        qk2 = torch.squeeze(qk2,dim=-1) #256,6,6
        qk2 = self.activation_function(qk2)
        self.qk2 = self.drop(qk2)
        self.alphas2 = F.softmax(self.qk2, dim=-1) #256,6,6
        alphas2 = torch.unsqueeze(self.alphas2,dim=-1)
        # normalize 2, to make sure att_value and ui are not in different shuliangji
        alphas2 = alphas2.expand_as(value) #256,6,6,64
        att_value2 = alphas2*value
        att_value2 = att_value2.sum(dim=-2)

        # aggregate two heads------------------
        att_value = (att_value1 + att_value2)/2
        ui = uid_0*iid_0 #256,64
        ui = torch.unsqueeze(ui, dim = -2) #256,1,64
        em_out = torch.cat((ui,att_value),dim = -2)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

class GAT_attention_single(nn.Module):
    # conduct graph attention network(GAT), single head
    def __init__(self,opt):
        super(GAT_attention_single, self).__init__()
        em_dim = opt.em_dim
        num_node = opt.num_node-1
        self.DEVICE = opt.DEVICE
        self.att_W = nn.Linear(em_dim,em_dim)
        self.att_a = nn.Linear(3*em_dim,1)
        self.layer_norm = nn.LayerNorm(em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        # normalize 1
        embeddings = self.layer_norm(embeddings)
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iid_0 = embeddings[:,1,:] #256,64
        iid = torch.unsqueeze(iid_0, dim = -2)
        iid = torch.unsqueeze(iid, dim = -2) #256,1,1,64
        iatt = embeddings[:,2:,:] #256,6,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,6,64
        # Calculating attention parameter------------------------------
        query = torch.unsqueeze(ua,dim=-2)
        key = torch.unsqueeze(ua,dim=-3)
        query = query.expand(query.size()[0],query.size()[1],query.size()[1],query.size()[-1])
        key = key.expand(key.size()[0],key.size()[2],key.size()[2],key.size()[-1])
        iid = iid.expand_as(key)
        # key: 256,6,6,64; query:256,6,6,64; iid: 256,6,6,64
        self.wq = self.att_W(query)
        self.wk = self.att_W(key)
        self.wiid = self.att_W(iid)
        qk = torch.cat((self.wq,self.wk,self.wiid),dim=-1) # 256,6,6,192
        qk = self.att_a(qk) #256,6,6,1
        qk = torch.squeeze(qk,dim=-1) #256,6,6
        qk = self.activation_function(qk)
        self.qk = self.drop(qk)
        self.alphas = F.softmax(self.qk, dim=-1) #256,6,6
        alphas = torch.unsqueeze(self.alphas,dim=-1)
        value = query * key
        # normalize 2, to make sure att_value and ui are not in different shuliangji
        value = self.layer_norm(value)
        alphas = alphas.expand_as(value) #256,6,6,64
        att_value = alphas*value
        att_value = att_value.sum(dim=-2)
        ui = uid_0*iid_0 #256,64
        ui = torch.unsqueeze(ui, dim = -2) #256,1,64
        em_out = torch.cat((ui,att_value),dim = -2)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out


class plus_attention(nn.Module):
    def __init__(self, opt):
        super(plus_attention, self).__init__()
        em_dim = opt.em_dim
        num_node = opt.num_node-1
        self.DEVICE = opt.DEVICE
        self.attention_act = opt.attention_act
        self.att_W_v = nn.Linear(em_dim,1)
        self.layer_norm = nn.LayerNorm(em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        query = embeddings[:,0,:]
        query = torch.unsqueeze(query, dim = -2)
        key = embeddings[:,1:,:]
        value = key
        q_pro = query
        k_pro = key
        qk = q_pro + k_pro
        if self.attention_act == 'tanh':
            qk = torch.tanh(qk)
        qk = self.att_W_v(qk)
        qk = torch.squeeze(qk, dim=-1)
        alphas = F.softmax(qk, dim=-1)
        alphas = torch.unsqueeze(alphas,dim=-1)
        self.alphas = alphas.expand_as(value)
        em_a = self.alphas*value
        em_out = torch.cat((query,em_a),dim = -2)
        return em_out

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

