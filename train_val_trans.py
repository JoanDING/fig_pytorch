import os
import time
import shutil
import torch
import numpy as np
import logging
import argparse
import torch.utils.data as data
import random

from eval_metrics.eval import *

from amazon_men import *
from model_trans import *


def train(opt):
  print('loading training data...')
  data_train = MenTrain(opt.data_path)

  
  data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)
  print('loading test data...')                                      
  data_test = MenTest(opt.data_path)
  data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=0,
                                                 collate_fn=collate_val)
  print('data done.')
  # for d in data_loader_train:
    # print(d[0].size())
    # print(d[0],'\n',d[1])
    # break
  print('data test done')
  
  opt.vocab_size = data_train.vocab_size
  opt.self_att = False
  opt.DEVICE = torch.device("cuda:"+str(opt.gpu) if torch.cuda.is_available() else "cpu")
  # opt.DEVICE = torch.device("cpu")
  print opt
  print('building model')
  model = Model(opt).to(opt.DEVICE)
  print('model built')
  tic = time.time()
  
  iter_cnt = 0
  dis_freq = 100
  eval_freq = 10
  
  for ep_id in range(opt.max_eps):
    
    for iter_id, train_batch in enumerate(data_loader_train):
      model.train()
      model.train_step(*train_batch)
      
      if iter_cnt%dis_freq==0:
        toc = time.time()
        print('Epoch: {}-{}, total iter: {}, loss: {}, time: {}'.format(ep_id,iter_id,iter_cnt,model.loss.data, toc-tic))
        tic = toc
      iter_cnt += 1 # place here for viewing the result of first iteration
      
    if ep_id%eval_freq==0:  
      print('Eval...Epoch {}'.format(ep_id))
      
      rank_res = {}
      eval_res = {}
      pre = []
      ndcg = []
      ap = []
      mrr = []
      rec = []
      for test_batch in data_loader_test:
        model.eval()
        scores = model.get_output(test_batch[0])
        # print('scores: type {} shape {}'.format(type(scores),scores.shape))
        uid = test_batch[3]
        targets = test_batch[1]
        candidates = test_batch[2]
        _, sort_idx = torch.sort(scores,descending=True)
        rank_list = [candidates[i] for i in sort_idx]
        rank_res[uid] = rank_list
        metrics = compute_user_metrics(rank_list, targets, opt.topk)
        pre.append(metrics[0])
        rec.append(metrics[1])
        ap.append(metrics[2])
        ndcg.append(metrics[3])
        mrr.append(metrics[4])
        # print('rank_list: type {} len {}'.format(type(rank_list),len(rank_list)))
      print('metrics type: {} values {}'.format(type(metrics),metrics))
      print('scores: Precision {}, Recall {}, mAP {}, NDCG {}, MRR {}'.format(np.mean(pre),np.mean(rec),np.mean(ap),np.mean(ndcg),np.mean(mrr),))
      print('rank_list top{}: {}'.format(opt.topk, rank_list[:opt.topk]))
      print('targets: {}'.format(targets))
    
        
    
    
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--em_dim', default=64, type=int,
                      help='embeddings dim.')
  parser.add_argument('--att_dim', default=64, type=int,
                      help='Attention transform dim.')
  parser.add_argument('--num_node', default=8, type=int,
                      help='number of nodes in graph.')
  parser.add_argument('--share_scorer', default=1, type=int,
                      help='flag for sharing scorers, 0 or 1')
  parser.add_argument('--classme', default=0, type=int,
                      help='flag for sharing scorers, 0 or 1')
  parser.add_argument('--margin', default=0.2, type=float,
                      help='Attention softmax temperature.')
  parser.add_argument('--topk', default=10, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--max_eps', default=500, type=int,
                      help='max epoches')                      
  parser.add_argument('--batch_size', default=256, type=int,
                      help='training and val batch size')
  parser.add_argument('--data_path', default='/home/binyi//amazon-men-group-cp', type=str,
                      help='root path of data')                     
  parser.add_argument('--lr', default=0.001, type=float,
                      help='initial learning rate')
  parser.add_argument('--gpu', default=0, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--dropout', default=0, type=float,
                      help='Attention softmax temperature.')
  parser.add_argument('--weight_decay', default=0, type=float,
                      help='Attention softmax temperature.')
  parser.add_argument('--checkpoint', default='', type=str,
                      help='Attention softmax temperature.')
                      
  opt = parser.parse_args()
  train(opt)
  
if __name__ == '__main__':
    main()
        
      
  
  