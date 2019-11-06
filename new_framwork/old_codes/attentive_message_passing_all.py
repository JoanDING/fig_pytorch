import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from base_att_module import *
import pdb


# ---------------------------------------------------------------------
# overall handle iid and iatt------------------------------------------
# ---------------------------------------------------------------------

class att0(nn.Module):
    def __init__(self,opt):
        super(att0, self).__init__()
        self.att_layer = Att_trans_cat(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
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
        ui_att_adj = ui + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out

class att0_16(nn.Module):
    def __init__(self,opt):
        super(att0_16, self).__init__()
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
        self.alphas,value = self.att_layer(i_em)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        att_value = self.mp_w(att_value)
        ui_att_adj = ui + att_value
        em_out = self.activation_function(ui_att_adj)
        em_out = self.drop(em_out)
        return em_out

class att0_1_multiply(nn.Module):
    def __init__(self,opt):
        super(att0_1_multiply, self).__init__()
        self.att_layer = Att_multiply(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
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

class att0_1(nn.Module):
    def __init__(self,opt):
        super(att0_1, self).__init__()
        self.att_layer = Att_cat(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        #em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        #em_out = em_out[:,1:,:]
        return em_out

class att0_11(nn.Module):
    def __init__(self,opt):
        super(att0_11, self).__init__()
        self.att_layer = Att_cat(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        em_out = em_out[:,1:,:]
        return em_out

class att0_12(nn.Module):
    def __init__(self,opt):
        super(att0_12, self).__init__()
        self.att_layer = Att_cat(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        self.alphas,value = self.att_layer(embeddings)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        em_out = em_out[:,1:,:]
        return em_out

class att0_13(nn.Module):
    def __init__(self,opt):
        super(att0_13, self).__init__()
        self.att_layer = Att_cat(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        self.alphas,value = self.att_layer(embeddings)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        em_out = em_out[:,1:,:]
        return em_out

class att0_14(nn.Module):
    def __init__(self,opt):
        super(att0_14, self).__init__()
        self.att_layer = Att_cat_withid(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        iid_0 = embeddings[:,1,:] #256,64
        iid = torch.unsqueeze(iid_0, dim = -2)
        iid = torch.unsqueeze(iid, dim = -2) #256,1,1,64
        self.alphas,value = self.att_layer(embeddings,iid)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

class att0_15(nn.Module):
    def __init__(self,opt):
        super(att0_15, self).__init__()
        self.att_layer = Att_cat_norm(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out



class att0_2(nn.Module):
    def __init__(self,opt):
        super(att0_2, self).__init__()
        self.att_layer = Att_cat_norm(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

class att0_3(nn.Module):
    def __init__(self,opt):
        super(att0_3, self).__init__()
        self.att_layer = Att_cat_norm(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

class att0_4(nn.Module):
    def __init__(self,opt):
        super(att0_4, self).__init__()
        self.att_layer = Att_cat_norm(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        self.alphas,value = self.att_layer(embeddings)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

class att0_5(nn.Module):
    def __init__(self,opt):
        super(att0_5, self).__init__()
        self.att_layer = Att_cat_norm_inte(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        self.alphas,value = self.att_layer(embeddings)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

class att0_6(nn.Module):
    def __init__(self,opt):
        super(att0_6, self).__init__()
        self.att_layer = Att_cat_withid_norm(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        iid_0 = embeddings[:,1,:] #256,64
        iid = torch.unsqueeze(iid_0, dim = -2)
        iid = torch.unsqueeze(iid, dim = -2) #256,1,1,64
        self.alphas,value = self.att_layer(embeddings,iid)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

class att0_7(nn.Module):
    def __init__(self,opt):
        super(att0_7, self).__init__()
        self.att_layer = Att_cat_withid_inte_norm(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        iid_0 = embeddings[:,1,:] #256,64
        iid = torch.unsqueeze(iid_0, dim = -2)
        iid = torch.unsqueeze(iid, dim = -2) #256,1,1,64
        self.alphas,value = self.att_layer(embeddings,iid)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

# 1-------------------------------------------------------------------
class att1_1(nn.Module):
    def __init__(self,opt):
        super(att1_1, self).__init__()
        self.att_layer = Att_trans_cat_inte(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        # normalize 1
        embeddings = self.layer_norm(embeddings)
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iatt = embeddings[:,1:,:] #256,6,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,6,64

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(ua)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        return em_out

class att1(nn.Module):
    def __init__(self,opt):
        super(att1, self).__init__()
        self.att_layer = Att_trans_cat(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        # normalize 1
        embeddings = self.layer_norm(embeddings)
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iatt = embeddings[:,1:,:] #256,6,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,6,64

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(embeddings)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        return em_out


# 2 -------------------------------------------------------------------
class att2(nn.Module):
    def __init__(self,opt):
        super(att2, self).__init__()
        self.att_layer = Att_trans_cat(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
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
        # key: 256,6,6,64; query:256,6,6,64; iid: 256,6,6,64
        self.alphas,value = self.att_layer(embeddings)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        ui = uid_0*iid_0 #256,64
        ui = torch.unsqueeze(ui, dim = -2) #256,1,64
        em_out = torch.cat((ui,att_value),dim = -2)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

# 2 -------------------------------------------------------------------
class att3(nn.Module):
    def __init__(self,opt):
        super(att3, self).__init__()
        self.att_layer = Att_trans_cat_withid(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

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
        self.alphas,value = self.att_layer(ua,iid)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        ui = uid_0*iid_0 #256,64
        ui = torch.unsqueeze(ui, dim = -2) #256,1,64
        em_out = torch.cat((ui,att_value),dim = -2)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

# without normalization for original embeddings
class att3_1(nn.Module):
    def __init__(self,opt):
        super(att3_1, self).__init__()
        self.att_layer = Att_trans_cat_withid(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        # normalize 1
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iid_0 = embeddings[:,1,:] #256,64
        iid = torch.unsqueeze(iid_0, dim = -2)
        iid = torch.unsqueeze(iid, dim = -2) #256,1,1,64
        iatt = embeddings[:,2:,:] #256,6,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,6,64

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(ua,iid)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        ui = uid_0*iid_0 #256,64
        ui = torch.unsqueeze(ui, dim = -2) #256,1,64
        em_out = torch.cat((ui,att_value),dim = -2)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

# involving interaction feature in calculating attetnion
class att3_3(nn.Module):
    def __init__(self,opt):
        super(att3_3, self).__init__()
        self.att_layer = Att_trans_cat_withid_inte(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

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
        self.alphas,value = self.att_layer(ua,iid)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        ui = uid_0*iid_0 #256,64
        ui = torch.unsqueeze(ui, dim = -2) #256,1,64
        em_out = torch.cat((ui,att_value),dim = -2)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

# without normalization for original embeddings annd value
class att3_4(nn.Module):
    def __init__(self,opt):
        super(att3_4, self).__init__()
        self.att_layer = Att_trans_cat_withid(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        # normalize 1
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iid_0 = embeddings[:,1,:] #256,64
        iid = torch.unsqueeze(iid_0, dim = -2)
        iid = torch.unsqueeze(iid, dim = -2) #256,1,1,64
        iatt = embeddings[:,2:,:] #256,6,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,6,64
        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(ua,iid)
        att_value = alphas*value
        att_value = att_value.sum(dim=-2)
        ui = uid_0*iid_0 #256,64
        ui = torch.unsqueeze(ui, dim = -2) #256,1,64
        em_out = torch.cat((ui,att_value),dim = -2)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

# without linear projection for query and key in calculating attention
class att3_5(nn.Module):
    def __init__(self,opt):
        super(att3_5, self).__init__()
        self.att_layer = Att_cat_withid(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

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
        self.alphas,value = self.att_layer(ua,iid)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        ui = uid_0*iid_0 #256,64
        ui = torch.unsqueeze(ui, dim = -2) #256,1,64
        em_out = torch.cat((ui,att_value),dim = -2)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

# 4 -------------------------------------------------------------------
# concat uid, iid and iattributes to construct the base feature for calculating attention

class att4(nn.Module):
    def __init__(self,opt):
        super(att4, self).__init__()
        self.att_layer = Att_trans_cat_withid(opt.activation_fun,opt.dropout, 2*opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)
        self.layer_norm2 = nn.LayerNorm(2*opt.em_dim)
        self.trans = nn.Linear(2*opt.em_dim,opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iid_0 = embeddings[:,1,:] #256,64
        ui0 = torch.cat((uid_0,iid_0),dim=-1) #256,128
        ui1 = torch.unsqueeze(ui0, dim = -2) #256,1,128
        ui = torch.unsqueeze(ui1, dim = -2) #256,1,1,128
        iatt = embeddings[:,2:,:] #256,6,64
        uid = uid.expand_as(iatt)
        ua = torch.cat((uid,iatt),dim=-1) #256,6,128

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(ua,ui)
        value = self.layer_norm2(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = torch.cat((ui1,att_value),dim = -2)
        em_out = self.trans(em_out)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out


class att4_1(nn.Module):
    # with interaction in calculating attention
    def __init__(self,opt):
        super(att4_1, self).__init__()
        self.att_layer = Att_trans_cat_withid_inte(opt.activation_fun,opt.dropout, 2*opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)
        self.layer_norm2 = nn.LayerNorm(2*opt.em_dim)
        self.trans = nn.Linear(2*opt.em_dim,opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iid_0 = embeddings[:,1,:] #256,64
        ui0 = torch.cat((uid_0,iid_0),dim=-1) #256,128
        ui1 = torch.unsqueeze(ui0, dim = -2) #256,1,128
        ui = torch.unsqueeze(ui1, dim = -2) #256,1,1,128
        iatt = embeddings[:,2:,:] #256,6,64
        uid = uid.expand_as(iatt)
        ua = torch.cat((uid,iatt),dim=-1) #256,6,128

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(ua,ui)
        value = self.layer_norm2(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = torch.cat((ui1,att_value),dim = -2)
        em_out = self.trans(em_out)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out

class att4_2(nn.Module):
    # with interaction in calculating attention
    def __init__(self,opt):
        super(att4_2, self).__init__()
        self.att_layer = Att_cat_withid_inte_norm(opt.activation_fun,opt.dropout, 2*opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)
        self.layer_norm2 = nn.LayerNorm(2*opt.em_dim)
        self.trans = nn.Linear(2*opt.em_dim,opt.em_dim)

    def forward(self, embeddings):
        embeddings = self.layer_norm(embeddings)
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iid_0 = embeddings[:,1,:] #256,64
        ui0 = torch.cat((uid_0,iid_0),dim=-1) #256,128
        ui1 = torch.unsqueeze(ui0, dim = -2) #256,1,128
        ui = torch.unsqueeze(ui1, dim = -2) #256,1,1,128
        iatt = embeddings[:,2:,:] #256,6,64
        uid = uid.expand_as(iatt)
        ua = torch.cat((uid,iatt),dim=-1) #256,6,128

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(ua,ui)
        value = self.layer_norm2(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = torch.cat((ui1,att_value),dim = -2)
        em_out = self.trans(em_out)
        em_out = self.activation_function(em_out)
        em_out = self.drop(em_out)
        return em_out
# 5 -------------------------------------------------------------------
class att5(nn.Module):
    # no normalization at the begining compared to att1
    def __init__(self,opt):
        super(att5, self).__init__()
        self.att_layer = Att_trans_cat(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)
        self.layer_norm = nn.LayerNorm(opt.em_dim)

    def forward(self, embeddings):
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iatt = embeddings[:,1:,:] #256,7,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,7,64

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(embeddings)
        value = self.layer_norm(value)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        return em_out

# 6 -------------------------------------------------------------------
class att6(nn.Module):
    # no normalization for value compared to att5
    def __init__(self,opt):
        super(att6, self).__init__()
        self.att_layer = Att_trans_cat(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iatt = embeddings[:,1:,:] #256,7,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,7,64

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(ua)
        att_value = self.alphas*value
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        return em_out

# 7 -------------------------------------------------------------------
class att7(nn.Module):
    # no normalization at the begining compared to att1
    def __init__(self,opt):
        super(att7, self).__init__()
        self.att_layer = Att_trans_sum(opt.activation_fun,opt.dropout,opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        uid_0 = embeddings[:,0,:] #256,64
        uid = torch.unsqueeze(uid_0, dim = -2) #256,1,64
        iatt = embeddings[:,1:,:] #256,7,64
        uid.expand_as(iatt)
        ua = uid*iatt #256,7,64

        # Calculating attention parameter------------------------------
        self.alphas,value = self.att_layer(ua)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        return em_out


class att8(nn.Module):
    def __init__(self,opt):
        super(att8, self).__init__()
        self.att_layer = AFM(opt.activation_fun, opt.dropout, opt.em_dim)
        if opt.activation_fun == 'leaky_relu':
            self.activation_function = nn.LeakyReLU()
        self.drop = nn.Dropout(opt.dropout)

    def forward(self, embeddings):
        self.alphas,value = self.att_layer(embeddings)
        att_value = self.alphas*value
        att_value = att_value.sum(dim=-2)
        em_out = att_value[:,1:,:]
        em_out = self.activation_function(em_out)
        #em_out = self.activation_function(att_value)
        em_out = self.drop(em_out)
        #em_out = em_out[:,1:,:]
        return em_out
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class GAT_attention_multi(nn.Module):
    # conduct graph attention network(GAT), multi head
    def __init__(self,opt):
        super(GAT_attention_multi, self).__init__()
        em_dim = opt.em_dim
        num_node = opt.num_node-1
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

class mul_attention(nn.Module):
    def __init__(self, opt):
        super(mul_attention, self).__init__()
        em_dim = opt.em_dim
        num_node = opt.num_node-1
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

class plus_attention(nn.Module):
    def __init__(self, opt):
        super(plus_attention, self).__init__()
        em_dim = opt.em_dim
        num_node = opt.num_node-1
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

