import torch
import torch.nn as nn
import torchvision.models as model
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class djj(nn.Module):
  def __init__(self, opt):
    em_dim = opt.embedding_dim
    self.recurrent = opt.recurrent
    num_update = opt.num_update
    num_em = opt.num_em
    self.attention = opt.attention
    
    att_dim = opt.att_dim
    
    
    self.em_layer = nn.Embedding(num_em, em_dim)
    if self.attention:
      if self.recurrent:
        self.cur_node = nn.Linear(em_dim, att_dim) # recurrent may use orthogonal initialization
        self.con_node = nn.Linear(em_dim, att_dim)
        self.mu_trans = nn.Linear(att_dim, 1)
        
        self.new_cur = nn.Linear(em_dim, em_dim) # here can be no bias, for context-aware updating that adds new_con as bias.
        self.new_con = nn.Linear(em_dim, em_dim)
      else:
        self.cur_node1 = nn.Linear(em_dim, att_dim)
        self.con_node1 = nn.Linear(em_dim, att_dim)
        self.mu_trans1 = nn.Linear(att_dim, 1)
        self.new_cur1 = nn.Linear(em_dim, em_dim)
        self.new_con1 = nn.Linear(em_dim, em_dim)
        
        
        self.cur_node2 = nn.Linear(em_dim, att_dim)
        self.con_node2 = nn.Linear(em_dim, att_dim)
        self.mu_trans2 = nn.Linear(att_dim, 1)
        self.new_cur2 = nn.Linear(em_dim, em_dim)
        self.new_con2 = nn.Linear(em_dim, em_dim)
        
    else:
      if self.recurrent:
        self.new_cur = nn.Linear(em_dim, em_dim)
        self.new_con = nn.Linear(em_dim, em_dim)
      else:
        self.new_cur1 = nn.Linear(em_dim, em_dim)
        self.new_con1 = nn.Linear(em_dim, em_dim)
        
        self.new_cur2 = nn.Linear(em_dim, em_dim)
        self.new_con2 = nn.Linear(em_dim, em_dim)
        
    self.score_layer = nn.Sequential(nn.Linear(8*em_dim, 2*em_dim), nn.ReLU(), nn.Linear(2*em_dim, 1), nn.Sigmoid())
      
  def forward_pass(self, batch_inds):
    ems = self.em_layer(batch_inds)
    
    
    
  def get_loss(batch_inds, batch_gts):
    pass
    
  
    
    