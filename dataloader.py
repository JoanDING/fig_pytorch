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

def collate_val(data):
  """
  Build mini-batch tensors from a list of tuples from getitem.
  """
  samps, iids, pos = zip(*data)
  print('samps\n{}'.format(samps.shape))
  print('samps type\n{}'.format(type(samps)))
  samps = torch.tensor(samps[0])
  # negs = torch.stack(negs)
  # print('poss\n{},{}'.format(type(poss),poss))
  # print('negs\n{},{}'.format(type(negs),negs))  
  return samps, poss, negs
  
  
def trans_idx(data_list, ids_map):
  data_array = np.array(data_list)
  data_array = np.transpose(data_array,[1,0])
  m,n = data_array.shape
  new_data = []
  for i in range(m):
    ids = data_array[i,:]
    id_map = ids_map[i]
    new_ids = []
    for ii in ids:
      new_ids.append(id_map[ii])
    new_data.append(new_ids)
  return new_data

  


class MenTrain(data.Dataset):
  def __init__(self, in_path, split='train'):
    super(MenTrain, self).__init__()
    data_file = os.path.join(in_path, 'amazon-men-group.{}'.format(split))
    indice_map_file = os.path.join(in_path, 'amazon-men-group.indice_map.json')
    item_file = os.path.join(in_path, 'amazon-men-group.item_map.json')
    
    print('Loading {} data from {}'.format(split,data_file))
    
    data_list = []
    item_list = {} # {iid:[attrs]}
    with open(data_file) as f:
      for l in f:
        l = l.strip().split()
        l = map(int, l) # if anything wrong at other place, uncomment this, and remove the ids_map file.
        iid_s = str(l[1])
        item_list[iid_s] = item_list.pop(iid_s, []) + [tuple(l[2:])]
        item_list[iid_s] = list(set(item_list[iid_s]))
        data_list.append(l)
    self.data_list = data_list
    
    ####add test set#####
    test_file = os.path.join(in_path, 'amazon-men-group.test')
    with open(test_file) as f:
      
    
    
    
    
    
    
    ####add test set#####
    
    len_cnt = 0
    for i, j in item_list.items():
      if len(j)>1:
        len_cnt+=1
    print('{} items with different attributes'.format(len_cnt))
    if not os.path.isfile(item_file):
      with open(item_file, 'w') as f:
        jstr = json.dumps(item_list)
        f.write(jstr)
      print('item-attributes map saved, in {}'.format(item_file))
      
    # transform all the uids, iids, and attributes ids to a uniform indices for embedding.
    print('transforming data')
    if os.path.isfile(indice_map_file):
      jdm = json.load(open(indice_map_file))
      ids_map = jdm['ids_map']
      cnt = jdm['cnt']
      new_data = trans_idx(data_list, ids_map)
    else:
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
      with open(indice_map_file,'w') as f:
        jstr = json.dumps({'ids_map':ids_map,'cnt':cnt})
        f.write(jstr)
      
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
    neg_file = os.path.join(in_path, 'amazon-men-group.{}.negatives'.format(split))
    item_file = os.path.join(in_path, 'amazon-men-group.item_map.json')
    print('Loading item-attributes map from {}'.format(item_file))
    jdi = json.load(open(item_file))
    item_map = {}
    for k,v in jdi.items():
      item_map[k] = list(v[0])
    
    self.item_map = item_map

    print('Loading {} data from {}'.format(split,data_file))

    # data_list = []
    pos_ids = {}
    uids = set()
    with open(data_file) as f:
      for l in f:
        l = l.strip().split()
        uids.add(l[0])
        uid = l[0]
        iid = l[1]
        pos_ids[uid] = pos_ids.pop(uid, []) + [iid]
        # data_list.append(l)
    # self.data_list = data_list 
    self.uids = list(uids)
    self.pos_ids = pos_ids
    
    print('Loaded')
    print('{} user ids in test set'.format(len(uids)))
    print('transforming indices...')
    indice_map_file = os.path.join(in_path, 'amazon-men-group.indice_map.json')
    print('Load indices map from {}'.format(indice_map_file))
    jdm = json.load(open(indice_map_file)) 
    
    ids_map = jdm['ids_map'] # list with 8 dicts, for 8 nodes elements.
    cnt = jdm['cnt']
    # new_data = trans_idx(data_list, ids_map)
    # print('transformed')
    # new_array = np.array(new_data)
    # new_array = np.transpose(new_array, [1,0])
    # self.data_array = new_array
    self.ids_map = ids_map
    print('Loading negative data from {}'.format(neg_file))
    
    neg_ids = {}
    with open(neg_file) as f:
      for l in f:
        l = l.strip().split()
        # l = map(int, l)
        neg_ids[str(l[0])] = l[1:]
    uid_iid = {}
    for k,v in pos_ids.items():
      uid_iid[k] = v + neg_ids[k]
        
    print('Loaded.')
    
    self.neg_ids = neg_ids
    self.uid_iid = uid_iid
    
    print('{} users to test'.format(len(uid_iid)))
    print('uids==uid_iid: {}'.format(len(uid_iid)==len(uid_s)))
    
    
    
  def __getitem__(self, index):
    # uid, iids = self.uid_iid[index] # type: string
    uid = self.uids[index]
    iids = self.uid_iid[uid]
    pos = self.pos_ids[uid]
    num_item = len(iids)
    uids = [uid]*num_item
    
    uid_iid_array = np.array([uids,iids])
    attrs = []
    for iid in iids:
      attrs.append(self.item_map[iid])
    attrs_array = np.array(attrs)
    attrs_array = np.transpose(attrs_array,[1,0])
    
    samps = np.concatenate([uid_iid_array, attrs_array])
    print('Verifying concatenate')
    print('[13911, 34007, 30, 12, 341, 10, 4, 25]')
    for i in samps:
      print(len(set(i)))
    print('Verifying end.')
    new_samps = self._trans_idx(samps, self.ids_map)
    new_array = np.array(new_samps)
    new_array = np.transpose(new_array,[1,0])
    return new_array, iids, pos
    
    
  def _trans_idx(self, samp_array, ids_map):
    # data_array = np.array(data_list)
    # data_array = np.transpose(data_array,[1,0])
    m,n = data_array.shape
    new_data = []
    for i in range(m):
      ids = data_array[i,:]
      id_map = ids_map[i]
      new_ids = []
      for ii in ids:
        new_ids.append(id_map[ii])
      new_data.append(new_ids)
    return new_data    
    
    
  def __len__(self):
    return len(self.uid_iid)











    