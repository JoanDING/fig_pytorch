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
import sys
sys.path.append('..')
from eval_metrics.eval import *


# from evaluation.leaveoneout.LeaveOneOutEvaluate import evaluate_by_loo
# from evaluation.foldout.FoldOutEvaluate import evaluate_by_foldout
# from util.Logger import logger1

from amazon_men import *
from amazon_women import *
from pog import *
from model_att import *
import time


def test(opt):
  def load_model(model,checkpoint_PATH):
    name = time.time()
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT)

  print('loading training data...')
  if opt.dataset == 'women':
    data_train = WomenTrain(opt.data_path)
    data_test = WomenTest(opt.data_path)
  elif opt.dataset == 'men':
    data_train = MenTrain(opt.data_path)
    data_test = MenTest(opt.data_path)
  elif opt.dataset == 'pog':
    data_train = PogTrain(opt.data_path)
    data_test = PogTest(opt.data_path)

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
  print('restore model from {}'.format(opt.ckpt))
  model = Model(opt).to(opt.DEVICE)
  checkpoint_PATH = opt.ckpt
  load_model(model,checkpoint_PATH)
  rank_res = {}
  eval_res = {}
  pre = []
  ndcg = []
  ap = []
  mrr = []
  rec = []
  for test_batch in data_loader_test:
    model.eval()
    scores, score_overall = model.get_output(test_batch[0], test_batch[3])
    attention_alpha = model.att_layer.alphas
    #qk = model.att_layer.qk
    #wq = model.att_layer.wq
    #wk = model.att_layer.wk
    uid = test_batch[4]
    targets = test_batch[1]
    candidates = test_batch[2]
    _, sort_idx = torch.sort(score_overall,descending=True)
    rank_list = [candidates[i] for i in sort_idx]
    rank_res[uid] = rank_list
    metrics = compute_user_metrics(rank_list, targets, opt.topk)
    pre.append(metrics[0])
    rec.append(metrics[1])
    ap.append(metrics[2])
    ndcg.append(metrics[3])
    mrr.append(metrics[4])
    pdb.set_trace()

  bestpre = np.mean(pre)
  bestrec = np.mean(rec)
  bestap = np.mean(ap)
  bestndcg = np.mean(ndcg)
  bestmrr = np.mean(mrr)

  print('Precision %.4f, Recall %.4f, mAP %.4f, NDCG %.4f, MRR %.4f'%(bestpre, bestrec, bestap, bestndcg, bestmrr))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--em_dim', default=64, type=int,
                      help='embeddings dim.')
  parser.add_argument('--save_path', default = './model/', type=str, help='path to save model')
  parser.add_argument('--save_epoch', default = 200, type=int, help='epochs to save model')
  parser.add_argument('--num_layer', default=1, type=int,
                      help='num_layer')
  parser.add_argument('--att_dim', default=64, type=int,
                      help='Attention transform dim')
  parser.add_argument('--type', default=101, type=int,
                      help='type of attention')
  parser.add_argument('--activation_fun', default='leaky_relu', type=str,
                      help='Activation function')
  parser.add_argument('--attention_act', default='tanh', type=str,
                      help='Activation function for attention')
  parser.add_argument('--share_scorer', default=0, type=int,
                      help='flag for sharing scorers, 0 or 1')
  parser.add_argument('--classme', default=0, type=int,
                      help='flag for sharing scorers, 0 or 1')
  parser.add_argument('--topk', default=10, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--max_eps', default=220, type=int,
                      help='max epoches')
  parser.add_argument('--batch_size', default=512, type=int,
                      help='training and val batch size')
  parser.add_argument('--dataset', default='men', type=str,
                      help='root path of data')
  parser.add_argument('--lr', default=0.002, type=float,
                      help='initial learning rate')
  parser.add_argument('--lr_att', default=0.002, type=float,
                      help='initial att learning rate')
  parser.add_argument('--gpu', default=0, type=int,
                      help='Attention softmax temperature.')
  parser.add_argument('--dropout', default=0.2, type=float,
                      help='Atention softmax temperature.')
  parser.add_argument('--weight_decay', default=1e-6, type=float,
                      help='weight_decay')
  parser.add_argument('--weight_decay_att', default=2e-6, type=float,
                      help='weight_decay_att')
  parser.add_argument('--ckpt', default='../model/10160_200_1575774722.38', type=str,
                      help='ckpt')

  opt = parser.parse_args()
  if opt.dataset == 'men':
      opt.num_node = 7
      opt.data_path = '/storage/yjding/djj_mask/amazon-men-group-cp_mask'
  elif opt.dataset == 'women':
      opt.num_node = 7
      opt.data_path = '/storage/yjding/djj_mask/amazon-women-group-cp_mask'
  elif opt.dataset == 'pog':
      opt.num_node = 6
      opt.data_path = '/storage/yjding/djj_mask/pog'

  test(opt)

if __name__ == '__main__':
    np.random.seed(2016)
    torch.manual_seed(2016)
    torch.cuda.manual_seed_all(2016)
    main()

