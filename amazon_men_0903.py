import torch.utils.data as data
import os
import torch
import random
import numpy as np
import scipy.io
import json
import copy


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


def read_samples(file_path):
  res = []
  with open(file_path) as f:
    for l in f:
      l = l.strip().split()
      res.append(l)
  return res


def convert_uid_iid_to_sample(uid_iid, items_map):
  """
  input: uid_iid: dict, {uid:[iid]}
  output: dict, {uid:[uid,iid,a1,a2,...]}
  """
  res = {}
  for uid, iids in uid_iid.items():
    res[uid] = []
    for iid in iids:
      samp = [uid,iid]
      attrs = items_map[iid]
      samp += attrs
      res[uid].append(samp)
  return res
    
    

def prepare_data(in_path):
  print('preparing data...')
  train_file = 'amazon-men-group.train'
  test_file = 'amazon-men-group.test'
  
  train_list = read_samples(os.path.join(in_path, train_file))
  test_list = read_samples(os.path.join(in_path, test_file))
  
  # all the ids are string.
  all_data = train_list + test_list
  
  items_map = {}
  pos_samp = {} # for generating negative training samples
  for d in all_data:
    uid = d[0]
    iid = d[1]
    items_map[iid] = d[2:]
    pos_samp[uid] = pos_samp.pop(uid, []) + [iid]
    pos_samp[uid] = list(set(pos_samp[uid]))
  
  
  neg_samp = {} # pool for negative samples, orig idx
  all_iids = set(items_map.keys())
  for uid, iids in pos_samp.items():
    neg_samp[uid] = list(all_iids - set(iids))
  
  cnt = []
  data_array = np.array(all_data).T
  num_node, num_samp = data_array.shape
  ids_map = []
  
  
  for i in range(num_node):
    ids = data_array[i,:]
    ids_set = set(list(ids))
    num_ids = len(ids_set)
    begin_idx = sum(cnt)
    cnt.append(num_ids)
    id_map = {}
    for idx, id in enumerate(ids_set):
      id_map[id] = idx + begin_idx
    ids_map.append(id_map)
  
  uid_idx = ids_map[0]
  iid_idx = ids_map[1]
  idx_uid = {}
  idx_iid = {}
  for uid, idx in uid_idx.items():
    idx_uid[str(idx)] = uid
  for iid, idx in iid_idx.items():
    idx_iid[str(idx)] = iid
  
  
  vocab_size = sum(cnt)
  print('cnt:{}'.format(cnt))
  
  def get_max(l):
    res = []
    for i in l:
      res.append(max(i))
    return res
  def get_min(l):
    res = []
    for i in l:
      res.append(min(i))
    return res
  new_train = trans_idx(train_list, ids_map) # return list, samples with new uniform indices for embedding
  new_array = np.array(new_train).T
  print('new_array shape: {}'.format(new_array.shape))
  print('new_train max num at each node: min {} max {}'.format(new_array.min(axis=0),new_array.max(axis=0)))
  print('new_train max num at each node: min {} max {}'.format(get_min(new_train),get_max(new_train)))
  new_train = list(new_array)
  new_train = map(list,new_train)
  
  # new_test = trans_idx(test_list, ids_map)
  
  #### transform test negative ####
  test_neg_file = os.path.join(in_path, 'amazon-men-group.test.negatives_by')
  
  if os.path.isfile(test_neg_file):
    neg_uid_iid = {} # orig idx, for eval.
    with open(test_neg_file) as f:
      for l in f:
        l = l.strip().split(',')
        uid = l[0]
        neg_uid_iid[uid] = l[1:]
  else:
    neg_uid_iid = {}
    for ts in test_list:
      uid = ts[0]
      neg_items = random.sample(neg_samp[uid],99)
      neg_uid_iid[uid] = neg_items
    with open(test_neg_file,'w') as f:
      for uid, iids in neg_uid_iid.items():
        all_ids = [uid]+iids
        print>>f,','.join(all_ids)
  # construct test with old inds, and then call trans_idx
  targets = {} # {uid: iid}, orig idx, for eval
  for ts in test_list:
    uid = ts[0]
    iid = ts[1]
    targets[uid] = targets.pop(uid,[]) + [iid]
  
  test_pos_neg = {} # {uid: [iids(pos+neg)]}, orig idx for eval
  for k, v in targets.items():
    test_pos_neg[k] = v + neg_uid_iid[k]
    
  test_samps = convert_uid_iid_to_sample(test_pos_neg, items_map) # orig idx
  new_test = {} # orig uid as key, new idx in values
  for uid, samp in test_samps.items():
    tmp = trans_idx(samp,ids_map)
    tmp_array = np.array(tmp)
    tmp_list = list(tmp_array.T)
    tmp_list = map(list,tmp_list)
    new_test[uid] = tmp_list
  print(len(tmp))
  print(len(tmp_list))
  
  #### end transform test negative ####
  
  # saving...
  print('saving...')
  train_new_all = {'data': new_train, 'neg_samp':neg_samp, 'pos_samp':pos_samp, 'items_map':items_map,
                   'idx_iid':idx_iid,'idx_uid':idx_uid,'uid_idx':uid_idx,'iid_idx':iid_idx}
  with open(os.path.join(in_path,'train_data_transformed.json'),'w') as f:
    jstr = json.dumps(train_new_all)
    f.write(jstr)
    
  test_new_all = {'data':new_test,'targets':targets,'candidates':test_pos_neg}
  with open(os.path.join(in_path,'test_data_transformed.json'),'w') as f:
    jstr = json.dumps(test_new_all)
    f.write(jstr)
  
  with open(os.path.join(in_path, 'amazon-men-group.indice_map.json'),'w') as f:
    jstr = json.dumps({'ids_map':ids_map,'cnt':cnt})
    f.write(jstr)

  print('data preparing done.')
  
  
  


class MenTrain(data.Dataset):
  def __init__(self, in_path, split='train'):
    super(MenTrain, self).__init__()
    data_file = os.path.join(in_path, '{}_data_transformed.json'.format(split))
    indice_map_file = os.path.join(in_path, 'amazon-men-group.indice_map.json')

    
    
    if not os.path.isfile(data_file):
      prepare_data('/home/binyi/amazon-men-group-cp')
    print('Loading {} data from {}'.format(split,data_file))
    jdd = json.load(open(data_file))
    
    self.data_list = jdd['data']
    self.neg_items = jdd['neg_samp']
    self.pos_items = jdd['pos_samp']
    self.items_map = jdd['items_map']
    self.idx_iid = jdd['idx_iid']
    self.idx_uid = jdd['idx_uid']
    self.uid_idx = jdd['uid_idx']
    self.iid_idx = jdd['iid_idx']
    
    jdm = json.load(open(indice_map_file))
    ids_map = jdm['ids_map']
    cnt = jdm['cnt']    
    
    self.ids_map = ids_map
    
    self.vocab_size = sum(cnt)
    print('cnt list: {}'.format(cnt))
    print('vocab_size: {}'.format(self.vocab_size))

  
    num_samp = len(self.data_list)
    print('{} samples in {} set'.format(num_samp, split))
    
    
  def _generate_neg(self, samp):
    neg_samp = []
    uid = samp[0]
    uid_orig = self.idx_uid[str(uid)]
    neg_origs = self.neg_items[str(uid_orig)]
    neg_orig = random.choice(neg_origs)
    neg_id = self.iid_idx[neg_orig]
    neg_attr = self.items_map[neg_orig]
    
    
    samp_orig = [uid_orig,neg_orig] + neg_attr
    samp_trans = []
    for i,idx in enumerate(samp_orig):
      samp_trans.append(self.ids_map[i][idx])
    # print('samp_orig: {}'.format(samp_orig))
    # print('samp_trans: {}'.format(samp_trans))
    # assert samp_trans[0]==uid
    # assert samp_trans[1]==neg_id
    
    return samp_trans
      
      
  def __getitem__(self, index):
    pos = self.data_list[index]
    neg = self._generate_neg(pos)
    return torch.tensor(pos), torch.tensor(neg)
    
  def __len__(self):
    return len(self.data_list)
    
def collate_val(data):
  """
  Build mini-batch tensors from a list of tuples from getitem.
  """
  samps, targets, candidates, uid = zip(*data) # return tuples, each tuple with length of batch_size
  # print('samps\n{}'.format(len(samps)))
  # print('samps type\n{}'.format(type(samps)))
  # print('targets\n{},{}'.format(type(targets),len(targets)))
  # print('candidates\n{},{}'.format(type(candidates),len(candidates)))
  # print('negs\n{},{}'.format(type(negs),negs))
  samps = torch.tensor(samps[0]) 
  # print('samps: type {}, shape {}'.format(type(samps),samps.shape))
  # negs = torch.stack(negs)
  # print('poss\n{},{}'.format(type(poss),poss))
  # print('negs\n{},{}'.format(type(negs),negs))  
  return samps, targets[0], candidates[0], uid[0]    
    
class MenTest(data.Dataset):
  def __init__(self, in_path, split='test'):
    super(MenTest, self).__init__()
    data_file = os.path.join(in_path, '{}_data_transformed.json'.format(split))

    print('Loading {} data from {}'.format(split,data_file))
    jdd = json.load(open(data_file))
    
    print('Loaded')
    self.data_dict = jdd['data']
    self.targets = jdd['targets']
    self.candidates = jdd['candidates']
    uids = self.data_dict.keys()
    print('number stat: data {}, targets {}, candidates {}'.format(len(self.data_dict),len(self.targets),len(self.candidates)))
    
    print('{} user ids in test set'.format(len(uids)))
    self.uids = uids
    
    
  def __getitem__(self, index):
    # uid, iids = self.uid_iid[index] # type: string
    uid = self.uids[index]
    usr_data = self.data_dict[uid]
    usr_targets = self.targets[uid]
    usr_candidates = self.candidates[uid]
    return usr_data, usr_targets, usr_candidates, uid
         
    
  def __len__(self):
    return len(self.uids)

    # print('preparing data...')
    # prepare_data('/home/binyi/amazon-men-group-cp')









    