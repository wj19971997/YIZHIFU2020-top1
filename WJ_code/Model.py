import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from WJ_code.seed import seed
import torch.nn.functional as F
from time import time
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, base_input_dim, op_emb_nuniq, trans_emb_nuniq, num_class=2, op_emb_size_lst=[],
                 trans_emb_size_lst=[]):
        super().__init__()
        self.op_emb_nunique = op_emb_nuniq
        self.trans_emb_nunique = trans_emb_nuniq

        # base
        base_output_emb = 64

        # op
        self.op_embbedding_lst = []
        for op_n, op_emb_size in zip(op_emb_nuniq, op_emb_size_lst):
            self.op_embbedding_lst.append(nn.Embedding(op_n, op_emb_size))
        lstm_op_h_size = 32 * 4
        self.lstm_op = nn.LSTM(input_size=sum(op_emb_size_lst) + len(op_emb_size_lst),
                               hidden_size=lstm_op_h_size,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True,
                               )

        # trans
        self.trans_embbedding_lst = []
        for trans_n, trans_emb_size in zip(trans_emb_nuniq, trans_emb_size_lst):
            self.trans_embbedding_lst.append(nn.Embedding(trans_n, trans_emb_size))
        lstm_trans_h_size = 32 * 4
        self.lstm_trans = nn.LSTM(input_size=sum(trans_emb_size_lst) + len(trans_emb_size_lst),
                                  hidden_size=lstm_trans_h_size,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True
                                  )

        dim = 512  # // 2
        self.final_layer = nn.Sequential(nn.Linear(base_input_dim + (lstm_op_h_size * 2 + lstm_trans_h_size * 2), dim),
                                         nn.BatchNorm1d(dim),
                                         nn.ReLU(),

                                         nn.Dropout(0.5),  # 0.3

                                         nn.Linear(dim, num_class)
                                         )

    def forward(self, base, op, trans, op_seq_len, trans_seq_len):
        base, op, trans = base.float(), op.long(), trans.long()

        # base
        # base_output = self.base_dense(base)

        # op
        new_op = []
        for i, op_emb_layer in enumerate(self.op_embbedding_lst):
            op_emb_layer.to(device)
            op_each = op[:, :, i]
            op_emb = op_emb_layer(op_each)
            new_op.append(op_emb)
        new_op.append(op[:, :, i + 1:].float())
        op_input = torch.cat(new_op, dim=2)
        op_input = torch.nn.utils.rnn.pack_padded_sequence(op_input, op_seq_len, batch_first=True, enforce_sorted=False)
        op_output, _ = self.lstm_op(op_input)
        op_output, _ = torch.nn.utils.rnn.pad_packed_sequence(op_output, batch_first=True)
        op_output = nn.MaxPool1d(op_output.size(1))(op_output.permute(0, 2, 1)).squeeze()

        # trans
        new_trans = []
        for i, trans_emb_layer in enumerate(self.trans_embbedding_lst):
            trans_emb_layer.to(device)
            trans_each = trans[:, :, i]
            trans_emb = trans_emb_layer(trans_each)  # [1]
            new_trans.append(trans_emb)
        new_trans.append(trans[:, :, i + 1:].float())
        trans_input = torch.cat(new_trans, dim=2)
        trans_input = torch.nn.utils.rnn.pack_padded_sequence(trans_input, trans_seq_len, batch_first=True,
                                                              enforce_sorted=False)
        trans_output, _ = self.lstm_trans(trans_input)
        trans_output, _ = torch.nn.utils.rnn.pad_packed_sequence(trans_output, batch_first=True)
        trans_output = nn.MaxPool1d(trans_output.size(1))(trans_output.permute(0, 2, 1)).squeeze()

        output = torch.cat([base, op_output, trans_output], dim=1)

        output = self.final_layer(output)

        return output