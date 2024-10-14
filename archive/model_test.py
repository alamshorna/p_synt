import math
import os
#from tempfile import TemporaryDirectory
#from typing import Tuple
from Bio import SeqIO
import numpy as np
import wandb
import seaborn
import matplotlib.pyplot as plt
import random

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset, DataLoader
import sklearn.preprocessing as skp

#from encoder import data_process, token_index_aa, token_index_codon, baseline, masking, encode_aa, encode_codon
chars_aa = 'ARNDCQEGHILKMFPSTWYVX'
chars_aa = 'CSTAGPDEQNHRKMILVWYFX'
token_index_aa = {amino_acid:index for (index, amino_acid) in enumerate(chars_aa)}

#encode synonyms
token_index_aa['O'], token_index_aa['U'], token_index_aa['B'], token_index_aa['Z'] = 11, 4, 20, 20

#added a "[PAD]" token separately in the token dictionary (as in proteinBERT)
extra_tokens = ['[START]', '[END]', '[PAD]', '[MASK]']
for i in range(len(extra_tokens)):
    token_index_aa[extra_tokens[i]] = len(chars_aa) + i

def encode_aa(sequence):
    """
    generates the encoding for an amino acid sequence
    input: sequence (str)
    output: encoding (list)
    """
    encoding = [token_index_aa['[START]']] + [token_index_aa[character] for character in sequence] + [token_index_aa['[END]']]
    return encoding

chars_codon = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789?@*'
token_index_codon = {amino_acid:index for (index, amino_acid) in enumerate(chars_codon)}

for i in range(len(extra_tokens)):
    token_index_codon[extra_tokens[i]] = len(chars_codon) + i

codon_mapping = {"TTT":"A", "TTC":"B", "TTA":"C", "TTG":"D",
    "TCT":"E", "TCC":"F", "TCA":"G", "TCG":"H",
    "TAT":"I", "TAC":"J", "TAA":"K", "TAG":"L",
    "TGT":"M", "TGC":"N", "TGA":"O", "TGG":"P",
    "CTT":"Q", "CTC":"R", "CTA":"S", "CTG":"T",
    "CCT":"U", "CCC":"V", "CCA":"W", "CCG":"X",
    "CAT":"Y", "CAC":"Z", "CAA":"a", "CAG":"b",
    "CGT":"c", "CGC":"d", "CGA":"e", "CGG":"f",
    "ATT":"g", "ATC":"h", "ATA":"i", "ATG":"j",
    "ACT":"k", "ACC":"l", "ACA":"m", "ACG":"n",
    "AAT":"o", "AAC":"p", "AAA":"q", "AAG":"r",
    "AGT":"s", "AGC":"t", "AGA":"u", "AGG":"v",
    "GTT":"w", "GTC":"x", "GTA":"y", "GTG":"z",
    "GCT":"0", "GCC":"1", "GCA":"2", "GCG":"3",
    "GAT":"4", "GAC":"5", "GAA":"6", "GAG":"7",
    "GGT":"8", "GGC":"9", "GGA":"?", "GGG":"@"}

amino = {"TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L",
    "TCT":"S", "TCC":"S", "TCA":"S", "TCG":"S",
    "TAT":"Y", "TAC":"Y", "TAA":"$", "TAG":"$",
    "TGT":"C", "TGC":"C", "TGA":"$", "TGG":"W",
    "CTT":"L", "CTC":"L", "CTA":"L", "CTG":"L",
    "CCT":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAT":"H", "CAC":"H", "CAA":"Q", "CAG":"Q",
    "CGT":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "ATT":"I", "ATC":"I", "ATA":"I", "ATG":"M",
    "ACT":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAT":"N", "AAC":"N", "AAA":"K", "AAG":"K",
    "AGT":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GTT":"V", "GTC":"V", "GTA":"V", "GTG":"V",
    "GCT":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAT":"D", "GAC":"D", "GAA":"E", "GAG":"E",
    "GGT":"G", "GGC":"G", "GGA":"G", "GGG":"G"}


def baseline(truthloaderlist, alphabet):
    #print(truthloaderlist)
    chars = chars_aa if alphabet == 'aa' else chars_codon
    frequencies = {i:0 for i in range(len(chars)+3)}
    current_batch_freq = {}
    for batch in truthloaderlist:
        batch = batch.tolist()
        #print(batch)
        current_batch_freq = {token:batch.count(token) for token in batch}
        for token in current_batch_freq:
            frequencies[token] += batch.count(token)
    return frequencies

def translation_window(sequence):
    """
    separate a raw DNA sequence into a codons
    to be inputted into encode_codon() function
    
    only includes the residues from [start:stop]
    input: DNA sequence (str)
    output: codon list (list)
    """
    start_location = sequence.find('ATG')
    answer = []
    for i in range(start_location, len(sequence), 3):
        current_residue = sequence[i:i+3]
        if current_residue in amino.keys() and amino[current_residue] != '$':
            answer.append(current_residue)
    return answer

def encode_codon(sequence):
    """
    generates the encoding for a codon sequence
    translation window determined by the first occurence of 'AUG'
    uses translation function defined in concatenation_script.py
    input: DNA sequence (str)
    output: encoding (list)
    """
    codon_list = translation_window(sequence)
    return [token_index_codon['[START]']] + [token_index_codon[codon_mapping[codon]] for codon in codon_list] + [token_index_codon['[END]']]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_dict(list):
    new_dict = {}
    for char in list:
        if char in new_dict.keys():
            new_dict[char] += 1
        else:
            new_dict[char] = 0
    return new_dict

#masking function
def masking(model, data, masking_proportion):
    """
    randomly chooses masking_proportion of tokens to cover
    returns data with indices masked
    """
    #keep pulling randomly until you get an equal number of them masked for all of the residues
    # presence = {key:False for key in range(20)}
    # for k in range(len(data)):
    #     if data[k]<20:
    #         presence[data[k].item()] = True
    # #print('presence', presence)
    # for key in presence:
    #     if not presence[key]:
    #         return data
    #print(data)
    data = list(data)
    amt_data = len(data)
    num_desired = int(masking_proportion*amt_data)
    chosen_indices = random.sample(list(range(amt_data)), num_desired)
    #print('chosen_indices', chosen_indices)
    masking_choices = random.choices(['null', 'correct', 'incorrect'], [.8, .1, .1], k=num_desired)
    #print(chosen_indices, masking_choices)
    for i in range(len(chosen_indices)):
        # if data[chosen_indices[i]].item() == 0:
        #     data[chosen_indices[i]] = token_index_aa['[MASK]']
        if data[chosen_indices[i]].item() == model.tokens["[PAD]"]:
            continue
        elif masking_choices[i] == 'null':
            data[chosen_indices[i]] = torch.tensor(model.tokens['[MASK]'])
        elif masking_choices[i] == 'correct':
            continue
        elif masking_choices[i] == 'incorrect':
            #rewrite to not include the correct amino acid
            data[chosen_indices[i]] = model.tokens[random.choice(list(model.tokens.keys()))]
    return data
    #print(count)
    #print(fully_masked)
    return data


def data_process(model, fasta, do_mask):
    """
    Performs complete data_preprocessing of amino acid sequences in a fasta_file
    including tokenization, concatenation, chunking, and padding
    input: file_path (str) containing protein sequences (and ids)
    output: set of pytorch tensors of batch_size, where the last is padded
    
    note that batch_size is inversely related to chunk_size
    """
    data_iterator = iter(list(SeqIO.parse(open(fasta), 'fasta')))
    tokenizer = encode_aa if model.alphabet == 'aa' else encode_codon
    if do_mask == True:
        tokenized_data = [masking(tokenizer(str(data.seq)), 0.15, model.alphabet) for data in data_iterator]
    else:
        tokenized_data = [tokenizer(str(data.seq)) for data in data_iterator]
    
    for i in range(len(tokenized_data)):
        this_len = len(tokenized_data[i])
        if this_len > model.max_length:
            tokenized_data[i] = tokenized_data[i][0:model.max_length]
        elif this_len < model.max_length:
            tokenized_data[i] = tokenized_data[i] + ([model.tokens["[PAD]"]] * (model.max_length-this_len))

    return tokenized_data


class data_set(Dataset):
    """
    data_set class created for dataloading
    simply calls the data_process function from encoder.py to initialize
    """
    def __init__(self, model, fasta, do_mask):
        self.data = data_process(model, fasta, do_mask)

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return self.data[index]
        

class PositionalEncoding(nn.Module):
    def __init__(self, fasta_file, max_length, alphabet_string, d_model):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_length).unsqueeze(1) #(max_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        size = pe.size()
        self.pe = pe #.reshape(size[2], size[0], size[1])

    def forward(self, embedding):
        embedding = embedding.unsqueeze(1)
        embedding = embedding + self.pe[:embedding.size(0)]
        return embedding.squeeze(1)
   

class DummyModel (nn.Module):
    def __init__(self, d_model, fasta_file, eval_file, alphabet_string, max_length = 512):
        super(DummyModel, self).__init__()
        self.model_type = 'Transformer'
        
        self.fasta_file = fasta_file
        self.eval_file = eval_file
        self.alphabet = alphabet_string

        #input information
        self.ntoken = len(token_index_aa) if alphabet_string == 'aa' else len(token_index_codon)
        self.max_length = max_length
        self.tokens = token_index_aa if alphabet_string == 'aa' else token_index_codon
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(fasta_file, self.max_length, self.alphabet, self.d_model)
        self.embedding = nn.Embedding(self.ntoken, self.d_model)
        self.tokenizer = encode_aa if alphabet_string == 'aa' else encode_codon
        self.cut = 20 if alphabet_string == 'aa' else 64
        
        # self.lstm = nn.LSTM(self.nimp, 512, 6, batch_first = True)
        encoder_layer = TransformerEncoderLayer(d_model, 8)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 6)
       
        decoder_layer = TransformerDecoderLayer(d_model, 8)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, 6)
        #(64 by ntokens) = average up all of the embeddings
        #dimensionality reduction
        #describe how similar the embeddings are
        #if that clusters we're doing well


        self.linear = nn.Linear(self.d_model, self.ntoken)
        self.epochs = 30
        self.log_interval = 1
        self.learning_rate = 0.01 
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        masked = src
        src = self.embedding(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)
        output = self.linear(src)
        output = nn.ReLU(src)
        output = self.linear(src)
        #set to zero for [PAD], [MASK]
        output[:, 27:] = -1e7 #make a hyperparameter
        output = torch.nn.functional.softmax(output)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, epoch):
    model.eval()
    eval_truth = data_set(model, model.eval_file, False).data
    eval_truth = torch.flatten(torch.tensor(eval_truth, dtype = torch.long))
    evalloader = DataLoader(eval_truth, model.max_length)
    masking_count = {token:0 for token in range(len(model.tokens))}
    with torch.no_grad():
        total_loss, count = 0, 0
        for sequence in evalloader:
            masked_sequence = torch.tensor(masking(model, sequence, 0.15))
            out = model(sequence)
            for k in range(len(masked_sequence)):
                current_letter = masked_sequence[k].item()
                true_letter = sequence[k].item()
                loss = model.loss_function(out[k], torch.tensor(true_letter))
                total_loss += loss
                count += 1
    return total_loss/count

def train(model):
    model.train()
    truth = data_set(model, model.fasta_file, False).data
    truth = torch.from_numpy(np.array(truth))
    truthloader = DataLoader(truth)
    epoch_number = 0
    for epoch in range(model.epochs):
        epoch_number += 1
        overall_loss = 0
        for sequence_batch in truthloader:
            for sequence in sequence_batch:
                total_loss, count = 0, 0
                masked_sequence = torch.tensor(masking(model, sequence, 0.15))
                out = model(sequence)
                for k in range(len(masked_sequence)):
                    current_letter, true_letter = masked_sequence[k], sequence[k]
                    loss = model.loss_function(out[k], true_letter)
                    total_loss += loss
                total_loss.backward()
                count += 1
                model.optimizer.step()
                model.scheduler.step()
                model.optimizer.zero_grad()
            overall_loss += total_loss
        if epoch % model.log_interval == 0 or epoch == 0:
            epoch_loss = str(overall_loss/(epoch_number*count))
            print("Epoch", epoch+1, "loss", epoch_loss)
            #print(out)
            val_loss = str(evaluate(model, epoch_number))
            #print("Validation Loss", val_loss)

test_model = DummyModel(64, 'data/micro_aa.fasta', 'data/micro_test_aa.fasta', 'aa', 512)
train(test_model)