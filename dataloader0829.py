import torch.utils.data as data
import os
import torch
import random
import numpy as np
import scipy.io
import json

RNG_SEED = 1234
random.seed(RNG_SEED)

def collate_fn(data):
  """
  Build mini-batch tensors from a list of tuples from getitem.
  """
  poss, negs = zip(*data)
  # print('poss\n{}'.format(poss))
  # print('negs\n{}'.format(negs))
  poss = torch.stack(poss)
  negs = torch.stack(negs)
  # print('poss\n{},{}'.format(type(poss),poss))
  # print('negs\n{},{}'.format(type(negs),negs))  
  return poss, negs
    

class MenTrain(data.Dataset):
  def __init__(self, in_path, split='train'):
    super(MenTrain, self).__init__()
    data_file = os.path.join(in_path, 'amazon-men-group.{}'.format(split))
    print('Loading {} data from {}'.format(split,data_file))
    
    data_list = []
    with open(data_file) as f:
      for l in f:
        l = l.strip().split()
        l = map(int, l)
        data_list.append(l)
    self.data_list = data_list
    
    # transform all the uids, iids, and attributes ids to a uniform indices for embedding.
    print('transforming data')
    data_array = np.array(data_list)
    data_array = np.transpose(data_array,[1,0])
    m,n = data_array.shape
    cnt = []
    ids_map = []
    new_data = []
    for i in range(m):
      ids = data_array[i,:]
      ids_set = set(list(ids))
      # ids_sort = sorted(list(ids_set))
      num_ids = len(ids_set)
      begin_idx = sum(cnt)
      cnt.append(num_ids)
      id_map = {}
      for i,j in enumerate(ids_set):
        id_map[j] = i + begin_idx
      ids_map.append(id_map)
      new_ids = []
      for ii in ids:
        new_ids.append(id_map[ii])
      new_data.append(new_ids)
      
    new_array = np.array(new_data)
    new_array = np.transpose(new_array, [1,0])
    self.data_array = new_array
    self.ids_map = ids_map
    
    self.vocab_size = sum(cnt)
    print('cnt list: {}'.format(cnt))
    print('vocab_size: {}'.format(self.vocab_size))
    print('transformed.')
    # end transform
    
    data_dict = {}
    item_dict = {}
    for d in new_array:
      uid, iid = d[0], d[1]
      uid = str(uid).strip()
      iid = str(iid).strip()
      data_dict[uid] = data_dict.pop(uid, []) + [iid]
      item_dict[iid] = d[2:]
    
    print('generating negative samples...')
    neg_array = self._generate_neg(new_array, item_dict, data_dict)
    print('generated.')
    self.neg_array = neg_array
    num_samp = len(data_list)
    print('{} samples in {} set'.format(num_samp, split))
    
    
  def _generate_neg(self, data_list, item_dict, data_dict):
    neg_list = []
    iids = set(item_dict.keys())
    
    for d in data_list:
      uid, iid = d[0], d[1]
      uid_s = str(uid).strip()
      iid_s = str(iid).strip()
      neg_item = iids - set(data_dict[uid_s])
      neg_iid = random.choice(list(neg_item))
      attrs = item_dict[neg_iid]
      neg_iid_int = int(neg_iid)
      neg_samp = [uid, neg_iid_int]+list(attrs)
      neg_list.append(neg_samp)
    neg_array = np.array(neg_list)
    return neg_array
      
      
  def __getitem__(self, index):
    pos = self.data_array[index]
    neg = self.neg_array[index]
    return torch.tensor(pos), torch.tensor(neg)
    
  def __len__(self):
    return len(self.data_list)
    
    
    
class MenTest(data.Dataset):
  def __init__(self, in_path, split='test'):
    super(MenTest, self).__init__()
    data_file = os.path.join(in_path, 'amazon-men-group.{}'.format(split))
    print('Loading {} data from {}'.format(split,data_file)) 
    
    data_list = []
    with open(data_file) as f:
      for l in f:
        l = l.strip().split()
        data_list.append(l)
    self.data_list = data_list    
    
  def __getitem__(self, index):
    pass
    
  def __len__(self):
    pass











    