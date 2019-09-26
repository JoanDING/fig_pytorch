import os
import json

def get_items(file_path):
  iids = set()
  with open(file_path) as f:
    for l in f:
      l = l.strip().split()
      iids.add(l[1])
  return iids
  
  
def comp(s1,s2):
  pass
  
tr_ids = get_items('/home/binyi/amazon-men-group-cp/amazon-men-group.train')
ts_ids = get_items('/home/binyi/amazon-men-group-cp/amazon-men-group.test')
val_ids = get_items('/home/binyi/amazon-men-group-cp/amazon-men-group.validation')
tr_val = tr_ids|val_ids

negs = set()
with open('/home/binyi/amazon-men-group-cp/amazon-men-group.test.negatives') as f:
  for l in f:
    l = l.strip().split(',')
    negs.update(set(l[1:]))

print('tr: {}, val: {}, ts: {}, neg: {}'.format(len(tr_ids),len(val_ids),len(ts_ids),len(negs)))
print('tr_ids-ts_ids: {}'.format(len(tr_ids-ts_ids)))
print('ts in tr: {}'.format(ts_ids<=tr_ids))
print('neg in tr: {}'.format(negs<=tr_ids))

print('ts in tr_val: {}'.format(ts_ids<=tr_val))
print('neg in tr_val: {}'.format(negs<=tr_val))