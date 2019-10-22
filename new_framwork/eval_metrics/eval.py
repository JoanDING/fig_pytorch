import Precision
import Recall
import MAP
import MRR
import NDCG
import pdb

def compute_user_metrics(rank_list,targets, topk=10):
  target_items = set(targets)
  rank_list = rank_list[:topk]
  Pre = Precision.getPre(rank_list, target_items)
  Rec = Recall.getRec(rank_list, target_items)
  ap = MAP.getAP(rank_list, target_items)
  dcg = NDCG.getNDCG(rank_list, target_items)
  rr = MRR.getMRR(rank_list, target_items)
  return Pre, Rec, ap, dcg, rr


def compute_metrics(rank_lists, targets):
  pass
