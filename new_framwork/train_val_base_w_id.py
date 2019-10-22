import os
import time
import shutil
import torch
import numpy as np
import logging
import argparse
import torch.utils.data as data
import random
import pdb
from eval_metrics.eval import *


# from evaluation.leaveoneout.LeaveOneOutEvaluate import evaluate_by_loo
# from evaluation.foldout.FoldOutEvaluate import evaluate_by_foldout
# from util.Logger import logger1

from amazon_men import *
from amazon_women import *
from model_base_w_id import *


def train(opt):
  print('loading training data...')
  if opt.dataset == 'women':
    data_path = '/storage/yjding/djj_mask/amazon-women-group-cp_mask'
    data_train = WomenTrain(data_path)
    data_test = WomenTest(data_path)
  elif opt.dataset == 'men':
    data_path = '/storage/yjding/djj_mask/amazon-men-group-cp_mask'
    data_train = MenTrain(data_path)
    data_test = MenTest(data_path)

  t1 = time.time()
  data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0,
                                            collate_fn=collate_fn)
  t2 = time.time()
  print('traindata loading time:', t2-t1)
  print('loading test data...')
  data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=0,
                                                 collate_fn=collate_val)
  print('data done.')
  # print('data test')
  # for d in data_loader_train:
    # print(d[0].size())
    # print(d[0],'\n',d[1])
    # break
  # print('data test done')

  opt.vocab_size = data_train.vocab_size
  opt.self_att = False
  opt.DEVICE = torch.device("cuda:"+str(opt.gpu) if torch.cuda.is_available() else "cpu")
  # opt.DEVICE = torch.device("cpu")
  print opt
  print('building model')
  model = Model(opt).to(opt.DEVICE)
  print('model built')
  iter_cnt = 0
  eval_freq = 10
  tic = time.time()
  bestpre = 0
  bestndcg = 0
  bestap = 0
  bestmrr = 0
  bestrec = 0
  for ep_id in range(opt.max_eps):
    t0 = time.time()
    loss_epoch = 0
    for iter_id, train_batch in enumerate(data_loader_train):
      model.train()
      model.train_step(*train_batch)
      iter_cnt += 1 # place here for viewing the result of first iteration
      loss_epoch += model.loss.data
    t1 = time.time()
    print('Epoch: {}, total iter: {}, loss: {}, time: {}'.format(ep_id,iter_cnt,loss_epoch/(iter_id+1.), t1-t0))
    #print('pos score: {}, neg_score: {}'.format(model.pos_sc[0], model.neg_sc[0]))
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
        scores = model.get_output(test_batch[0], test_batch[3])
        uid = test_batch[4]
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
      #print('metrics type: {} values {}'.format(type(metrics),metrics))
      print('scores: Precision %.4f, Recall %.4f, mAP %.4f, NDCG %.4f, MRR %.4f'%(np.mean(pre),np.mean(rec),np.mean(ap),np.mean(ndcg),np.mean(mrr)))
      if np.mean(ndcg) > bestndcg:
        best_epoch = ep_id
        bestpre = np.mean(pre)
        bestrec = np.mean(rec)
        bestap = np.mean(ap)
        bestndcg = np.mean(ndcg)
        bestmrr = np.mean(mrr)
      t2 = time.time()
      print('best epoch %d: Precision %.4f, Recall %.4f, mAP %.4f, NDCG %.4f, MRR %.4f'%(best_epoch, bestpre, bestrec, bestap, bestndcg, bestmrr))
      print('evaluate time: {}').format(t2-t1)
      #print('rank_list top{}: {}'.format(opt.topk, rank_list[:opt.topk]))
      #print('targets: {}'.format(targets))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--em_dim', default=64, type=int,
                      help='embeddings dim.')
  parser.add_argument('--num_layer', default=1, type=int,
                      help='num_layer')
  parser.add_argument('--att_dim', default=64, type=int,
                      help='Attention transform dim')
  parser.add_argument('--activation_fun', default='leaky_relu', type=str,
                      help='Activation function')
  parser.add_argument('--num_node', default=7, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--share_scorer', default=0, type=int,
                      help='flag for sharing scorers, 0 or 1')
  parser.add_argument('--classme', default=0, type=int,
                      help='flag for sharing scorers, 0 or 1')
  parser.add_argument('--topk', default=10, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--max_eps', default=500, type=int,
                      help='max epoches')
  parser.add_argument('--batch_size', default=256, type=int,
                      help='training and val batch size')
  parser.add_argument('--dataset', default='men', type=str,
                      help='root path of data')
  parser.add_argument('--lr', default=0.002, type=float,
                      help='initial learning rate')
  parser.add_argument('--gpu', default=0, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--dropout', default=0.2, type=float,
                      help='Atention softmax temperature.')
  parser.add_argument('--weight_decay', default=1e-6, type=float,
                      help='weight_decay')
  parser.add_argument('--checkpoint', default='', type=str,
                      help='checkpoint')

  opt = parser.parse_args()
  train(opt)

if __name__ == '__main__':
    np.random.seed(2019)
    torch.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    main()




