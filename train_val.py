import os
import time
import shutil
import torch
import numpy as np
import logging
import argparse
import torch.utils.data as data
import random

from dataloader import *
from model import *


def train(opt):
  print('loading training data...')
  data_train = MenTrain(opt.data_path)
  
  data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)
                                        
  print('data done.')
  print('data test')
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
  
  for ep_id in range(opt.max_eps):
    
    for iter_id, train_batch in enumerate(data_loader_train):
      model.train()
      model.train_step(*train_batch)
      
      if iter_cnt%dis_freq==0:
        toc = time.time()
        print('Epoch: {}-{}, total iter: {}, loss: {}, time: {}'.format(ep_id,iter_id,iter_cnt,model.loss.data, toc-tic))
        tic = toc
      iter_cnt += 1 # place here for viewing the result of first iteration
      
    print('Eval...Epoch {}'.format(ep_id))
    
        
    
    
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--em_dim', default=64, type=int,
                      help='embeddings dim.')
  parser.add_argument('--att_dim', default=64, type=int,
                      help='Attention transform dim.')
  parser.add_argument('--num_node', default=8, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--share_scorer', default=1, type=int,
                      help='flag for sharing scorers, 0 or 1')
  parser.add_argument('--classme', default=0, type=int,
                      help='flag for sharing scorers, 0 or 1')
  parser.add_argument('--margin', default=0.2, type=float,
                      help='Attention softmax temperature.')
  parser.add_argument('--topk', default=3, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--max_eps', default=500, type=int,
                      help='max epoches')                      
  parser.add_argument('--batch_size', default=256, type=int,
                      help='training and val batch size')
  parser.add_argument('--data_path', default='/home/binyi//amazon-men-group-cp', type=str,
                      help='root path of data')                     
  parser.add_argument('--lr', default=0.0001, type=float,
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
        
      
  
  