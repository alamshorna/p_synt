import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from Bio import SeqIO
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

from encoder import data_process, token_index_aa, token_index_codon

class data_set(Dataset):
    def __init__(self, fasta_file, batch_size, alphabet, do_mask):
        self.data = data_process(fasta_file, batch_size, alphabet, do_mask)
        #self.ground_truth = data_process(fasta_file, batch_size, alphabet, False)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return self.data[index]
    
#     def get(self, index):
#         return self.data[index]




# def data_embedding(fasta_file, d_model, alphabet_string):
#     data = data_process(fasta_file, d_model, alphabet_string, True)
#     embedding = nn.Embedding(len(token_index_codon), 1)
#     size = embedding(data).size()
#     data_embedding = embedding(data)
#     return data_embedding

#maximum_length
class PositionalEncoding(nn.Module):
    def __init__(self, fasta_file, d_model, alphabet_string):
        super(PositionalEncoding, self).__init__()
        #indices = indices_encoding(fasta_file, d_model, alphabet_string)
        max_length = d_model #data_process(fasta_file, d_model, alphabet_string, True).size()[1]
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        size = pe.size()
        self.pe = pe.reshape(size[2], size[0], size[1])

    def forward(self, embedding):
        embedding = embedding.t().unsqueeze(2)
        embedding = embedding + self.pe[:embedding.size(0)]
        return embedding.squeeze(2).t()
    
def number_sequences(fasta_file):
    fasta_data = list(SeqIO.parse(open(fasta_file), 'fasta'))
    return len(fasta_file)

class TransformerModel (nn.Module):
    def __init__(self, d_model, fasta_file, alphabet_string):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        self.fasta_file = fasta_file
        self.d_model = d_model
        self.alphabet = alphabet_string
        self.pos_encoder = PositionalEncoding(fasta_file, self.d_model, self.alphabet)
        #figure out inputs
        encoder_layer = TransformerEncoderLayer(d_model, 1)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 1)
        self.ninp = number_sequences(fasta_file)
        self.ntoken = len(token_index_aa) if alphabet_string == 'aa' else len(token_index_codon)
        self.embedding = nn.Embedding(self.ntoken, 512)
        self.decoder = nn.Linear(self.d_model, self.ntoken)
        self.init_weights()
        self.satya = 'satya'

        self.epochs = 30
        
        self.log_interval = 1
        self.learning_rate = 0.01
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, eval_fasta, epoch):
    model.eval()
    total_loss = 0
    eval_data = data_set(eval_fasta, 512, model.alphabet, True)
    true_data = data_set(eval_fasta, 512, model.alphabet, False)
    data_list, truth_list = list(enumerate(DataLoader(eval_data, 512))), list(enumerate(DataLoader(true_data, 512)))
    #print(data_list, truth_list)
    loss_function = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for i in range(len(data_list)):
            batch_values, truth_values = data_list[i][1], truth_list[i][1]
            out = model(batch_values)
            loss = loss_function(out, truth_values)
            total_loss += loss
            losses.append(loss.item())
        #print("Epoch: {} -> val loss: {}".format(epoch+1, total_loss/(len(data_list))*epoch+1))
        print("Actual val loss", np.mean(losses), total_loss, (len(data_list))*epoch+1)
    return None

def train(model, eval_fasta):
    model.train()
    total_loss = 0
    data, true = data_set(model.fasta_file, 512, model.alphabet, True), data_set(model.fasta_file, 512, model.alphabet, False)
    dataloaderlist, truthloaderlist = list(enumerate(DataLoader(data, 512))), list(enumerate(DataLoader(true, 512)))
    start_time = time.time()
    print('start train')
    for epoch in range(model.epochs):
        losses = []
        i = 0
        for j in range(len(dataloaderlist)):
            batch_values, truth_values = dataloaderlist[j][1], truthloaderlist[i][1]
            out = model(batch_values)
            i+=1
            loss = model.loss_function(out, truth_values)
            # loss = loss/512
            # print('train loss', loss)
            
            #print(loss.item())
            losses.append(loss.item())# += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            if i%100 == 0:
                print('done !')
                model.optimizer.step()
                model.optimizer.zero_grad()
        print(i)
        # print(losses)
        evaluate(model, eval_fasta, epoch)
        model.train()
        if epoch % model.log_interval == 0 or epoch == 0:
            print("Epoch: {} -> loss: {}".format(epoch+1, np.mean(losses)))
    #evaluate(model, eval_fasta, 30)

test_model =TransformerModel(512, 'data/mini_aa.fasta', 'aa')
# eval_data, true_data = data_process('data/mini_test.fasta', 512, 'codon', True), data_process('data/mini_test.fasta', 512, 'codon', False)
# print('eval', eval_data.size(), 'true', true_data.size())
eval_fasta = 'data/mini_test_aa.fasta'
train(test_model, eval_fasta)
# evaluate(test_model, 'data/mini.fasta')