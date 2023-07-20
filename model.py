import math
import os
#from tempfile import TemporaryDirectory
#from typing import Tuple
from Bio import SeqIO
import numpy as np
import wandb
import seaborn
import matplotlib.pyplot as plt


import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

from encoder import data_process, token_index_aa, token_index_codon, baseline, masking, encode_aa, encode_codon

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
    
def number_sequences(fasta_file):
    fasta_data = list(SeqIO.parse(open(fasta_file), 'fasta'))
    return len(fasta_file)

class TransformerModel (nn.Module):
    def __init__(self, d_model, fasta_file, eval_file, alphabet_string, max_length = 512):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        self.fasta_file = fasta_file
        self.eval_file = eval_file
        self.alphabet = alphabet_string

        #input information
        self.ninp = number_sequences(fasta_file)
        self.ntoken = len(token_index_aa) if alphabet_string == 'aa' else len(token_index_codon)
        self.max_length = max_length
        self.tokens = token_index_aa if alphabet_string == 'aa' else token_index_codon
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(fasta_file, self.max_length, self.alphabet, self.d_model)
        self.tokenizer = encode_aa if alphabet_string == 'aa' else encode_codon
        self.cut = 20 if alphabet_string == 'aa' else 64
        
        # self.lstm = nn.LSTM(self.nimp, 512, 6, batch_first = True)
        encoder_layer = TransformerEncoderLayer(d_model, 8)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 6)
        self.embedding = nn.Embedding(self.ntoken, self.d_model)
        self.decoder = nn.Linear(self.d_model, self.ntoken)
        self.epochs = 20
        self.log_interval = 1
        self.learning_rate = 0.00001 
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.init_weights()
        print(self.tokens)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        #print(src.shape)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        #add softmax
        output = torch.nn.functional.softmax(output)
        return output

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#should really rewrite this part so that it doesn't require storing the entire dictionary and instead just sums from inside of the function
def prob_dist_avg(lst, tokens, count):
    """
    place holder function to be replaced by some fancy numpy nonsense once I have wifi
    simply takes a list of probability distribution and performs and index-wise average
    """
    
    if len(lst) == 0:
        return np.array([0] * len(tokens))
    average = [0 for index in range(len(lst[0]))]
    for distribution in lst:
        for index in range(len(distribution)):
            #if count_dict[index] != 0:
            average[index] += distribution[index]#/count_dict[index]
            # else:
            #     average[index] += distribution[index]
    count = 1 if count == 0 else count
    average = [total/(len(lst)*count) for total in average]
    return np.array(average)

def evaluate(model, epoch, baseline_frequencies):
    model.eval()
    eval_truth = data_set(model, model.eval_file, False).data
    eval_truth = torch.flatten(torch.tensor(eval_truth, dtype = torch.long))
    evalloader = DataLoader(eval_truth, model.max_length)
    replacement_distributions = {token: np.array([0]*model.ntoken)  for token in range(len(model.tokens))}
    masking_count = {token:0 for token in range(len(model.tokens))}
    with torch.no_grad():
        total_loss, count = 0, 0
        for sequence in evalloader:
            masked_sequence = torch.tensor(masking(model, sequence, 0.15))
            out = model(sequence)
            for k in range(len(masked_sequence)):
                current_letter = masked_sequence[k].item()
                true_letter = sequence[k].item()
                if current_letter == model.tokens["[MASK]"]:
                    masking_count[true_letter] += 1
                    dist = nn.functional.softmax(out[k]).tolist()
                    #replacement_distributions[true_letter].append(dist)
                    replacement_distributions[true_letter] = np.add(replacement_distributions[true_letter], dist)
                loss = model.loss_function(out[k], torch.tensor(true_letter))
                total_loss += loss
                count += 1
    masking_count = {index:masking_count[index]+1 if masking_count[index]==0 else masking_count[index] for index in masking_count.keys()}
    masking_array = np.array([masking_count[key] for key in masking_count.keys()])
    replacement_distributions = np.array([np.divide(replacement_distributions[key], masking_count[key]) for key in replacement_distributions.keys()])
    replacement_distributions = np.array([np.divide(row, masking_array) for row in replacement_distributions])

    replacement_distributions = replacement_distributions[:model.cut, :model.cut]

    if model.alphabet == 'codon':
        replacement_distributions = np.delete(replacement_distributions, [10, 11, 14], axis=0)
        replacement_distributions = np.delete(replacement_distributions, [10, 11, 14], axis=1)

    replacement_distributions = np.transpose(replacement_distributions)
    replacement_distributions = np.array([np.array(nn.functional.softmax(torch.Tensor(dist))) for dist in replacement_distributions])
    replacement_distributions = np.transpose(replacement_distributions)
    replacement_distributions = np.array([np.array(nn.functional.softmax(torch.Tensor(dist))) for dist in replacement_distributions])

    # column_sums = replacement_distributions.sum(axis=0)
    # replacement_distributions = replacement_distributions/column_sums[None,:]
    # row_sums = replacement_distributions.sum(axis=1)
    # replacement_distributions = replacement_distributions/row_sums[:,None]
    # column_average = replacement_distributions.mean(axis=0)
    # replacement_distributions = replacement_distributions/column_average[None,:]
    # column_sums = np.array([])
    # replacement_distributions = [np.divide(replacement_distributions[i], masking_count[i]) for i in range(len(replacement_distributions))]

    print(replacement_distributions)
    for row in replacement_distributions:
        print(np.sum(row))
    # test_file_path = ''
    np.savetxt('test.csv', replacement_distributions, delimiter=',', fmt='%s')
    plt.clf()
    seaborn.heatmap(replacement_distributions)
    path = 'picture' + str(epoch) + '.png'
    plt.savefig(path)
    return total_loss/count
    #replacement_distributions = [prob_dist_avg(replacement_distributions[key], model.tokens, masking_count[key]) for key in replacement_distributions]
    #replacement_distributions = np.array([np.array(torch.Tensor(dist)).tolist() for dist in replacement_distributions])
    #replacement_distributions = np.transpose(np.array(replacement_distributions))
    # for i in range(len(replacement_distributions)):
    #     replacement_distributions[i] = nn.functional.softmax(torch.Tensor(dist))
    #replacement_distributions = np.transpose(replacement_distributions)

def train(model):
    model.train()
    truth = data_set(model, model.fasta_file, False).data
    truth = torch.flatten(torch.tensor(truth, dtype = torch.long))
    print(truth.size())
    truthloader = DataLoader(truth, model.max_length)
    print(baseline(list(truthloader), model.alphabet))
    epoch_number = 0
    for epoch in range(model.epochs):
        epoch_number += 1
        total_loss, count = 0, 0
        for sequence in truthloader:
            print(count)
            masked_sequence = torch.tensor(masking(model, sequence, 0.15))
            out = model(masked_sequence)
            #print(out.size())
            loss = model.loss_function(out, sequence)
            total_loss += loss
            count += 1
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
        if epoch % model.log_interval == 0 or epoch == 0:
            epoch_loss = str(total_loss/count)
            val_loss = str(evaluate(model, epoch_number, baseline(list(truthloader), model.alphabet)))
            print("Epoch", epoch+1, "loss", epoch_loss)
            print("Validation Loss", val_loss)
            # wandb.log({"loss": float(epoch_loss), "val loss": float(val_loss)})
            out_file =  open('data/last_run.txt', 'a')
            #loss_string = "Epoch " + str(epoch+1) + ": loss " + epoch_loss + " val loss " + val_loss + "\n"
            #out_file.write(loss_string)
            out_file.close()

# wandb.login()

test_model =TransformerModel(64, 'data/mini_codon.fasta', 'data/mini_test_codon.fasta', 'codon', 512)

# run = wandb.init(
#     # Set the project where this run will be logged
#     name = "transformer-model-human-aa-07_04_23-alamshorna",
#     project= "nucleotide",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": test_model.learning_rate,
#         "epochs": test_model.epochs,
#     })

#clear the out file, add the experiment name at the top
#out_file_name = "data/last_run.txt"
#experiment_name = "transformer-model-human-aa-07_04_23-alamshorna"
#clears the current contents of the file
# open(out_file_name, 'w').close()
# out_file = open(out_file_name, 'w')
# out_file.write(experiment_name + "\n")
# out_file.close()

train(test_model)