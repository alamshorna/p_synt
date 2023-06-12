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
    data = data_process(fasta_file, d_model, alphabet_string)
    embedding = nn.Embedding(len(token_index_codon), 1)
    size = embedding(data).size()
    data_embedding = embedding(data)
    #.reshape([size[0], size[1]])
    return data_embedding

#maximum_length

def positional_encoding(fasta_file, d_model, alphabet_string):
    indices = indices_encoding(fasta_file, d_model, alphabet_string)
    max_length = data_process(fasta_file, d_model, alphabet_string).size()[1]
    position = torch.arange(max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_length, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    size = pe.size()
    return pe.reshape(size[2], size[0], size[1])

# def positional_encoding(fasta_file, d_model, alphabet_string):
#     """
#     generate the embeddings and the positional encodings for data
#     sum to produce inputs for the transformer layers
#     """
#     data = data_embedding(fasta_file, d_model, alphabet_string)
#     positional = positional_function(fasta_file, d_model, alphabet_string)
#     return torch.add(data, positional)
def number_sequences(fasta_file):
    fasta_data = list(SeqIO.parse(open(fasta_file), 'fasta'))
    return len(fasta_file)

class TransformerModel (nn.Module):
    def __init__(self, d_model, fasta_file, nimp, alphabet_string):
        self(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = positional_encoding(fasta_file)
        #figure out inputs
        encoder_layer = TransformerEncoderLayer(d_model, 1)
        self.encoder = TransformerEncoder(encoder_layer, 1)
        self.nimp = number_sequences(fasta_file)
        self.embedding = data_embedding(fasta_file, d_model, alphabet_string)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self):
        src = self.embedding(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
    
# def train(model, dataloader):

# def predict(model, input):




# print(data_embedding('data/10K_codons_test.fasta', 512, 'codon').size())
# print(positional_encoding('data/10K_codons_test.fasta', 512, 'codon').size())
# print(word_embeddings('data/10K_codons_test.fasta', 512, 'codon'))