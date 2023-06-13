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
    #.reshape([size[0], size[1]])
    return data_embedding

#maximum_length
class PositionalEncoding(nn.Module):
    def __init__(self, fasta_file, d_model, alphabet_string):
    #def positional_encoding(fasta_file, d_model, alphabet_string):
        super(PositionalEncoding, self).__init__()
        indices = indices_encoding(fasta_file, d_model, alphabet_string)
        max_length = data_process(fasta_file, d_model, alphabet_string, True).size()[1]
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        size = pe.size()
        #print('pe', size)
        self.pe = pe.reshape(size[2], size[0], size[1])
        # print('pe', self.pe.size())

    def forward(self, embedding):
        embedding = embedding.t().unsqueeze(2)
        embedding = embedding + self.pe[:embedding.size(0)]
        print('pe', self.pe.size())
        print('embedding_forward', embedding.size())
        #print(self.pe.size(), embedding.size(0), self.pe[:embedding.size(0)])
        #print(embedding.size(0))
        print(self.pe[:embedding.size(0)].size())
        #print(embedding.size())
        return embedding.squeeze(2).t()
    
def number_sequences(fasta_file):
    fasta_data = list(SeqIO.parse(open(fasta_file), 'fasta'))
    return len(fasta_file)

class TransformerModel (nn.Module):
    def __init__(self, d_model, fasta_file, alphabet_string):
        super(TransformerModel, self).__init__()
        #self(TransformerModel, self).__init__()
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
        # print('embedding', src.size())
        src = self.pos_encoder(src)
        # print('positional_encoding', src.size())
        output = self.transformer_encoder(src)
        # print('output', output)
        # print('output', output.size())
        output = self.decoder(output)
        return output
    
# ntokens = len()  # size of vocabulary
# emsize = 200  # embedding dimension
# d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
# nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
# nhead = 2  # number of heads in ``nn.MultiheadAttention``
# dropout = 0.2  # dropout probability
# model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

import time

# criterion = nn.CrossEntropyLoss()
# lr = 5.0  # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model: nn.Module):
    model.train()
    epochs = 100
    total_loss = 0
    log_interval = 5
    start_time = time.time()
    learning_rate = 5.0
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    data = data_process(model.fasta_file, model.d_model, model.alphabet, do_mask = True)
    correct_data = data_process(model.fasta_file, model.d_model, model.alphabet, do_mask = False)
    start_time = time.time()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    for epoch in range(epochs):
        i = 0
        for batch in data:
            optimizer.zero_grad()
            #print(model)
            
            out = model(batch)
            print(i)
            print(batch)
            print(out)
            i+=1
            loss = loss_function(out, batch)
            print(loss)
            total_loss += loss

            optimizer.zero_grad()
            #loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            print(loss)
            optimizer.step()

            
            # if i % log_interval == 0 and i > 0:
            #     lr = scheduler.get_last_lr()[0]
            #     ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            #     cur_loss = total_loss / log_interval
            #     print(cur_loss)
            #     ppl = math.exp(cur_loss)
            #     print(f'| epoch {epoch:3d} | '
            #         f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
            #         f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            #     total_loss = 0
            #     start_time = time.time()
        
        if epoch % log_interval == 5 or epoch == 0:
            print("Epoch: {} -> loss: {}".format(epoch+1, total_loss/(len(data)*epoch+1)))

def evaluate(model: nn.Module, eval_data: Tensor):
    model.eval()
    out = model(input.to(device))
    out = out.topk(1).indices.view(-1)
    return out

test_model =TransformerModel(512, 'data/10K_codons_test.fasta', 'codon')
train(test_model)