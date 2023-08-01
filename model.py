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

index_token_aa = {index:aa for (aa, index) in token_index_aa.items()}

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

index_token_codon = {index:codon for (codon, index) in token_index_codon.items()}

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

mask_token_index = token_index_aa['[MASK]']
masking_proportion = 0.15 

#masking function
def masking(tokenized_sequence, masking_proportion):
    masked_sequence = tokenized_sequence.clone()
    num_tokens_to_mask = int(masking_proportion * len(tokenized_sequence))
    masked_indices = torch.randperm(len(tokenized_sequence))[:num_tokens_to_mask]
    masked_sequence[masked_indices] = mask_token_index  # Replace with [MASK] token
    return masked_sequence

def data_process(model, fasta):
    """
    Performs complete data_preprocessing of amino acid sequences in a fasta_file
    including tokenization, concatenation, chunking, and padding
    input: file_path (str) containing protein sequences (and ids)
    output: set of pytorch tensors of batch_size, where the last is padded
    """
    data_iterator = iter(list(SeqIO.parse(open(fasta), 'fasta')))
    tokenizer = encode_aa if model.alphabet == 'aa' else encode_codon
    tokenized_data = [tokenizer(str(data.seq)) for data in data_iterator]
    
    for i in range(len(tokenized_data)):
        this_len = len(tokenized_data[i])
        if this_len > model.max_length:
            tokenized_data[i] = tokenized_data[i][0:model.max_length]
        elif this_len < model.max_length:
            tokenized_data[i] = tokenized_data[i] + ([model.tokens["[PAD]"]] * (model.max_length-this_len))

    return tokenized_data


class Data_Set(Dataset):
    """
    data_set class created for dataloading
    simply calls the data_process function from encoder.py to initialize
    """
    def __init__(self, model, fasta):
        self.data = data_process(model, fasta)

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
        # embedding = embedding.unsqueeze(1)
        embedding = embedding + self.pe[:embedding.size(0)]
        return embedding.squeeze(1)
    
def number_sequences(fasta_file):
    fasta_data = list(SeqIO.parse(open(fasta_file), 'fasta'))
    return len(fasta_data)

class TransformerModel (nn.Module):
    def __init__(self, d_model, fasta_file, eval_file, alphabet_string, max_length = 512):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.fasta_file = fasta_file
        self.eval_file = eval_file
        self.alphabet = alphabet_string
        self.ninp = number_sequences(fasta_file)
        self.ntoken = len(token_index_aa) if alphabet_string == 'aa' else len(token_index_codon)
        self.max_length = max_length
        self.tokens = token_index_aa if alphabet_string == 'aa' else token_index_codon
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(fasta_file, self.max_length, self.alphabet, self.d_model)
        self.embedding = nn.Embedding(self.ntoken, self.d_model)
        self.tokenizer = encode_aa if alphabet_string == 'aa' else encode_codon
        self.cut = 20 if alphabet_string == 'aa' else 64
        encoder_layer = TransformerEncoderLayer(d_model, 8, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 6)
        #(64 by ntokens) = average up all of the embeddings
        #dimensionality reduction
        #describe how similar the embeddings are
        #if that clusters we're doing well


        self.linear = nn.Linear(self.d_model, self.ntoken)
        self.epochs = 30
        self.batch_size = 1
        self.log_interval = 1
        self.learning_rate = 0.05
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.init_weights()
        print(self.tokens)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        encoded = self.transformer_encoder(src)
        output = self.linear(encoded)
        #set to zero for [PAD], [MASK]
        output[:, 27:] = -1e7 #make a hyperparameter
        return torch.nn.functional.softmax(output)

def evaluate(model, epoch):
    model.eval()
    eval_truth = Data_Set(model, model.eval_file).data
    eval_truth = torch.from_numpy(np.array(eval_truth))
    evalloader = DataLoader(eval_truth)

    replacement_distributions = {token: np.array([0]*model.ntoken)  for token in range(len(model.tokens))}
    masking_count = {token:0 for token in range(len(model.tokens))}

    with torch.no_grad():
        total_loss, count = 0, 0
        for batch in evalloader:
            batch = batch.to(model.device)
            for sequence in batch:
                masked_sequence = torch.tensor(masking(sequence, 0.15))
                out = model(sequence)
                for k in range(len(masked_sequence)):
                    current_letter, true_letter = masked_sequence[k].item(), sequence[k].item()
                    if current_letter == model.tokens["[MASK]"]:
                        masking_count[true_letter] += 1
                        loss = model.loss_function(out[k], torch.tensor(true_letter))
                        total_loss += loss
                        dist = nn.functional.softmax(out[k]).tolist()
                        replacement_distributions[true_letter] = np.add(replacement_distributions[true_letter], dist)
                count += 1
    del evalloader
    
    masking_count = {index:masking_count[index]+1 if masking_count[index]==0 else masking_count[index] for index in masking_count.keys()}
    replacement_distributions = np.array([np.divide(replacement_distributions[key], masking_count[key]) for key in replacement_distributions.keys()])
    replacement_distributions = replacement_distributions[:model.cut, :model.cut]

    plt.clf()
    seaborn.heatmap(replacement_distributions[:20, :20])
    path = 'picture' + str(epoch) + '.png'
    plt.savefig(path)
    save_path = 'saved_model' + str(epoch) + '.pt'

    if epoch % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': total_loss/count,
            }, save_path)
    return total_loss

def train(model):
    model.train()
    truth = Data_Set(model, model.fasta_file).data
    truth = torch.from_numpy(np.array(truth))
    truthloader = DataLoader(truth, 32)
    for epoch in range(model.epochs):
        epoch_loss = 0
        for sequence_batch in truthloader:
            batch_loss = 0
            out = model(sequence_batch)
            print(out)
            batch_loss = model.loss_function(torch.transpose(out, 1, 2), sequence_batch)
            batch_loss.backward()
            model.optimizer.step()
            model.scheduler.step()
            model.optimizer.zero_grad()
            epoch_loss += batch_loss
            
        if epoch % model.log_interval == 0 or epoch == 0:
            print("Epoch", epoch+1, "loss", str(epoch_loss))
            last_sequence = out[-1]
            print(last_sequence, last_sequence.size())
            out_index = torch.argmax(last_sequence, dim=1)
            print(out_index)
            index_token_aa[25], index_token_aa[26], index_token_aa[27], index_token_aa[28] = '[START]', '[END]', '[MASK]', '[PAD]'
            out_token = [index_token_aa[index.item()] for index in out_index[0]]
            print("".join(out_token))
            print(list(SeqIO.parse(open(model.fasta_file), 'fasta'))[-1].seq)
            # val_loss = str(evaluate(model, epoch+1))
            # print("Validation Loss", val_loss)
            #wandb.log({"loss": float(epoch_loss), "val loss": float(val_loss)})

# wandb.login()

test_model = TransformerModel(64, 'data/micro_aa.fasta', 'data/micro_test_aa.fasta', 'aa', 512)

# run = wandb.init(
#     # Set the project where this run will be logged
#     name = "transformer-model-human-aa-07_04_23-alamshorna",
#     project= "nucleotide",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": test_model.learning_rate,
#         "epochs": test_model.epochs,
#     })

train(test_model)