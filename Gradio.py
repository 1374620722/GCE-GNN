import gradio as gr
import torch
from torchvision import transforms
import requests
from PIL import Image
import time
import argparse
import pickle
from model import *
from utils import *


def handle_data2(inputData):
  len_data = len(inputData)
  max_len = len_data
  # reverse the sequence
  us_pois = list(reversed(inputData)) + [0] * (max_len - len_data) if len_data < max_len else list(
    reversed(inputData[-max_len:]))

  us_msks = [1] * len_data + [0] * (max_len - len_data) if len_data < max_len else [1] * max_len
  return us_pois, us_msks, max_len


def Data_dell(data):
  inputs, mask, max_len = handle_data2(data)
  input = np.asarray(inputs)
  mask = np.asarray(mask)
  length = len(data)
  node = np.unique(input)
  items = node.tolist() + (max_len - len(node)) * [0]
  adj = np.zeros((max_len, max_len))
  for i in np.arange(len(input) - 1):
    u = np.where(node == input[i])[0][0]
    adj[u][u] = 1
    if input[i + 1] == 0:
      break
    v = np.where(node == input[i + 1])[0][0]
    if u == v or adj[u][v] == 4:
      continue
    adj[v][v] = 1
    if adj[v][u] == 2:
      adj[u][v] = 4
      adj[v][u] = 4
    else:
      adj[u][v] = 2
      adj[v][u] = 3
  alias_inputs = [np.where(node == i)[0][0] for i in input]

  return [torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),
          torch.tensor(mask), torch.tensor(input)]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall/sample')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=2)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')

opt = parser.parse_args()

num_node = 43098
opt.n_iter = 1
opt.dropout_gcn = 0.2
opt.dropout_local = 0.0
train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
if opt.validation:
  train_data, valid_data = split_validation(train_data, opt.valid_portion)
  test_data = valid_data
else:
  test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
train_data = Data(train_data)
test_data = Data(test_data)

adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))
model = model.eval()


def predict(inp):
  lst = list(map(int, inp.split(",")))
  alias_inputs, adj, items, mask, inputs = Data_dell(lst)
  alias_inputs = trans_to_cuda(alias_inputs).long()
  items = trans_to_cuda(items).long()
  adj = trans_to_cuda(adj).float()
  mask = trans_to_cuda(mask).long()
  inputs = trans_to_cuda(inputs).long()
  alias_inputs = alias_inputs.unsqueeze(0)
  items = items.unsqueeze(0)
  adj = adj.unsqueeze(0)
  mask = mask.unsqueeze(0)
  inputs = inputs.unsqueeze(0)
  hidden = model(items, adj, mask, inputs)  # 输出全局图和局部图得到的item Embedding
  get = lambda index: hidden[index][alias_inputs[index]]
  seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
  scores = model.compute_scores(seq_hidden, mask)
  sub_scores = scores.topk(20)[1]
  sub_scores = sub_scores.flatten().tolist()
  str_ = ','.join(str(i) for i in sub_scores)
  return str_

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()



