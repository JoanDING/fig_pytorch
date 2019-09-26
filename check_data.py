import os
import pdb
import json
import torch.utils.data as data

def get_items(file_path):
  pdb.set_trace()
  iids = set()
  with open(file_path) as f:
    for l in f.readlines():
      l = l.strip().split()
      iids.add(l[1])
  return iids

def read_json(json_file):
  jdd = json.load(open(json_file))

def check_idx(data_list, ids_map, cnt):
  pass

def comp(s1,s2):
  pass

train_file = 'amazon-men-group-cp_mask/train_data_transformed.json'
data = read_json(train_file)
pdb.set_trace()
f = open(train_file,'r')
for line in f.readlines():
    dic = json.loads(line)

    pdb.set_trace()
tr_ids = get_items('amazon-men-group-cp_mask/train_data_transformed.json')
pdb.set_trace()
ts_ids = get_items('amazon-men-group-cp_mask/test_data_transformed.json')
val_ids = get_items('amazon-men-group-cp_mask/amazon-men-group.indice_map.json')
tr_val = tr_ids|val_ids

negs = set()
with open('amazon-men-group-cp_mask/amazon-men-group.test.negatives') as f:
  for l in f:
    l = l.strip().split(',')
    negs.update(set(l[1:]))

print('tr: {}, val: {}, ts: {}, neg: {}'.format(len(tr_ids),len(val_ids),len(ts_ids),len(negs)))
print('tr_ids-ts_ids: {}'.format(len(tr_ids-ts_ids)))
print('ts in tr: {}'.format(ts_ids<=tr_ids))
print('neg in tr: {}'.format(negs<=tr_ids))

print('ts in tr_val: {}'.format(ts_ids<=tr_val))
print('neg in tr_val: {}'.format(negs<=tr_val))
