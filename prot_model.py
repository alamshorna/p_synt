import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from Bio import SeqIO

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from encoder import data_process, indices_encoding, token_index_aa, token_index_codon

def data_embedding(fasta_file, d_model, alphabet_string):
    data = data_process(fasta_file, d_model, alphabet_string, True)
    embedding = nn.Embedding(len(token_index_codon), 1)
    size = embedding(data).size()
    data_embedding = embedding(data)
    return data_embedding

#maximum_length
class PositionalEncoding(nn.Module):
    def __init__(self, fasta_file, d_model, alphabet_string):
        super(PositionalEncoding, self).__init__()
        indices = indices_encoding(fasta_file, d_model, alphabet_string)
        max_length = data_process(fasta_file, d_model, alphabet_string, True).size()[1]
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        size = pe.size()
        self.pe = pe.reshape(size[2], size[0], size[1])

    def forward(self, embedding):
        embedding = embedding.t().unsqueeze(2)
        # print(embedding.size())
        # print('pe:  ',self.pe[:embedding.size(0)].size())
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
        #self.decoder = nn.Linear(hidden_dimensions, output)
        self.decoder = nn.Linear(self.d_model, self.ntoken)
        self.init_weights()
        self.satya = 'satya'

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

def evaluate(model: nn.Module, eval_data, true_data, loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_number in range(eval_data.size(1)):
            # print(batch_number)
            # print(eval_data[:,batch_number].size())
            out = model(eval_data[:,batch_number])
            loss = loss_function(eval_data[batch_number], true_data[batch_number])
            total_loss += loss
    return total_loss/(batch_number+1)

def train(model: nn.Module, eval_data, true_data):
    model.train()
    epochs = 30
    total_loss = 0
    log_interval = 1
    start_time = time.time()
    learning_rate = 0.1
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    data = data_process(model.fasta_file, model.d_model, model.alphabet, do_mask = True)
    correct_data = data_process(model.fasta_file, model.d_model, model.alphabet, do_mask = False)
    start_time = time.time()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    for epoch in range(epochs):
        i = 0
        for batch in data:
        # print('data size: ', data.size())
        # print('data size 1:', data.size(1))
        # for batch_number in range(data.size(1)):
            # batch = data[:, batch_number]
            # print('train batch size:', batch.size())
            out = model(batch)
            i+=1
            loss = loss_function(out, batch)
            total_loss += loss

            #optimizer.zero_grad()
            loss = loss/512
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            #print(i)
            if i%512 == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # print(evaluate(model, eval_data, true_data, loss_function))

        if epoch % log_interval == 0 or epoch == 0:
            print("Epoch: {} -> loss: {}".format(epoch+1, total_loss/(len(data)*epoch+1)))

test_model =TransformerModel(512, 'data/mini.fasta', 'codon')
eval_data, true_data = data_process('data/mini_test.fasta', 512, 'codon', True), data_process('data/mini_test.fasta', 512, 'codon', False)
# print('eval', eval_data.size(), 'true', true_data.size())
train(test_model, eval_data, true_data)
# evaluate(test_model, 'data/mini.fasta')