import math
import os
from Bio import SeqIO
import numpy as np
import wandb
import seaborn
import matplotlib.pyplot as plt
import random
import sys

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import Dataset, DataLoader
import sklearn.preprocessing as skp
import umap

chars_aa = 'AGCSTPDEQNHRKMILVWYFX'
token_index_aa = {amino_acid:index for (index, amino_acid) in enumerate(chars_aa)}
token_index_aa['O'], token_index_aa['U'], token_index_aa['B'], token_index_aa['Z'] = 11, 4, 20, 20

extra_tokens = ['[START]', '[END]', '[PAD]', '[MASK]']
for i in range(len(extra_tokens)):
    token_index_aa[extra_tokens[i]] = len(chars_aa) + i

print(token_index_aa)

def encode_aa(sequence):
    """
    generates the encoding for an amino acid sequence
    input: sequence (str)
    output: encoding (list)
    """
    encoding = [token_index_aa['[START]']] + [token_index_aa[character] for character in sequence] + [token_index_aa['[END]']]
    return encoding

sample_protein = 'CSSSSTIFW'
print(encode_aa(sample_protein))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_process(fasta, max_length = 512):
    """
    Performs complete data_preprocessing of amino acid sequences in a fasta_file
    including tokenization, concatenation, chunking, and padding
    input: file_path (str) containing protein sequences (and ids)
    output: set of pytorch tensors of batch_size, where the last is padded
    """
    data_iterator = iter(list(SeqIO.parse(open(fasta), 'fasta')))
    tokenizer = encode_aa
    tokenized_data = [tokenizer(str(data.seq)) for data in data_iterator]
    for i in range(len(tokenized_data)):
        this_len = len(tokenized_data[i])
        if this_len > max_length:
            tokenized_data[i] = tokenized_data[i][0:max_length]
        elif this_len < max_length:
            tokenized_data[i] = tokenized_data[i] + ([token_index_aa["[PAD]"]] * (max_length-this_len))
    return torch.tensor(tokenized_data, dtype = torch.long)


def masking(tokenized_tensor, masking_proportion, mask_token_index, batch_size = 1, max_len = 512):
    masked_tensor = tokenized_tensor.clone()
    ignore_tensor = tokenized_tensor.clone()
    total_pad = torch.unique(masked_tensor, return_counts = True)[1][-1].item()
    unpadded_len = len(tokenized_tensor[0]) - total_pad
    num_tokens_to_mask = int(masking_proportion * unpadded_len)
    mask_tensors = []
    for i in range(masked_tensor.size(0)):
        valid_indices = (masked_tensor[i] != 23) & (torch.arange(max_len) != 0)
        valid_indices_to_mask = torch.arange(512)[valid_indices][1:]

        # Get random indices to mask from the valid indices
        mask_indices = torch.randperm(valid_indices_to_mask.size(0))[:num_tokens_to_mask]
        mask_indices = valid_indices_to_mask[mask_indices]

        # Set the value at the masked indices to 24
        masked_tensor[i, mask_indices] = mask_token_index

        # For the special masked tensor, set unmasked values to -100
        unmasked_indices = torch.ones(512, dtype=bool)
        unmasked_indices[mask_indices] = 0
        ignore_tensor[i, unmasked_indices] = -100
        ignore_tensor[i, mask_indices] = tokenized_tensor[i, mask_indices].clone()
    return masked_tensor, ignore_tensor

class Data_Set(Dataset):
    """
    data_set class created for dataloading
    simply calls the data_process function from encoder.py to initialize
    """
    def __init__(self, fasta):
        self.data = data_process(fasta)

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return self.data[index]
    

class LSTMModel(nn.Module):
    def __init__(self, d_model, fasta_file, eval_file, alphabet_string, max_length=512):
        super(LSTMModel, self).__init__()
        self.model_type = 'LSTM'
        self.fasta_file = fasta_file
        self.eval_file = eval_file
        self.alphabet = alphabet_string

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ntoken = len(token_index_aa) #if alphabet_string == 'aa' else len(token_index_codon)
        self.max_length = max_length
        self.tokens = token_index_aa #if alphabet_string == 'aa' else token_index_codon
        self.d_model = d_model

        self.embedding = nn.Embedding(self.ntoken, self.d_model, device=self.device)
        self.lstm = nn.LSTM(self.d_model, self.d_model, batch_first=True, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(self.d_model * 2, self.d_model, batch_first=True, bidirectional=True)
        self.lstm_layer3 = nn.LSTM(self.d_model * 2, self.d_model, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(2 * self.d_model, self.ntoken, device=self.device)

        self.epochs = 300
        self.batch_size = 32
        self.log_interval = 1
        self.learning_rate = 0.009 
        self.loss_function = nn.CrossEntropyLoss(size_average=True, ignore_index=-100)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.9)
        self.init_weights()
        print(self.tokens)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -math.sqrt(1/self.d_model), math.sqrt(1/self.d_model))
        #self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear.weight)
        #self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src)
        output, _ = self.lstm(src)
        output, _ = self.lstm_layer2(output)
        #output, _ = self.lstm_layer3(output)
        output = self.linear(output)
        return output
    
def evaluate(model, epoch):
    model.eval()
    eval_truth = Data_Set(model.eval_file).data
    eval_truth = torch.from_numpy(np.array(eval_truth))
    evalloader = DataLoader(eval_truth, model.batch_size)

    replacement_distributions = {token: torch.zeros(model.ntoken) for token in range(model.ntoken)}
    masking_count = {token:0 for token in range(model.ntoken)}
    for key, value in replacement_distributions.items():
        replacement_distributions[key] = replacement_distributions[key].to(device)
    with torch.no_grad():
        total_loss =  0
        for batch in evalloader:
            batch = batch.to(model.device)
            batch_loss = 0
            masked_batch, ignore_tensor = masking(batch, 0.15, model.tokens['[MASK]'])
            out = model(masked_batch)
            batch_loss = model.loss_function(torch.transpose(out, 1, 2), ignore_tensor)
            for i in range(len(batch)):
                for j in range(len(batch[i])):
                    current_token = masked_batch[i][j].item()
                    true_token = batch[i][j].item()
                    if current_token == model.tokens["[MASK]"]:
                        masking_count[batch[i][j].item()] += 1
                        replacement_distributions[true_token] = torch.add(replacement_distributions[current_token], out[i][j])
            total_loss += batch_loss
    masking_count = {index:masking_count[index]+1 if masking_count[index]==0 else masking_count[index] for index in masking_count.keys()}
    replacement_distributions = np.array([np.array(torch.divide(replacement_distributions[key], masking_count[key]).cpu()) for key in replacement_distributions.keys()])
    #replacement_distributions = np.array([np.array(nn.functional.softmax(torch.Tensor(dist))) for dist in replacement_distributions])

    print(replacement_distributions)
    plt.clf()
    np.savetxt('unnormalized.csv', replacement_distributions[:20, :20])
    seaborn.heatmap(replacement_distributions[:20, :20])
    path = 'picture' + str(epoch) + '.png'
    print('saving figure')
    plt.savefig(path)

    save_path = 'saved_model' + str(epoch) + '.pt'
    if epoch % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': total_loss/len(evalloader),
            }, save_path)
    return total_loss/len(evalloader)

def train(model):
    model.train()
    truth = Data_Set(model.fasta_file).data
    truth = torch.from_numpy(np.array(truth))
    print(truth, truth.size())
    truthloader = DataLoader(truth, model.batch_size)
    count = 0
    for epoch in range(model.epochs):
        epoch_loss = 0
        print(len(truthloader))
        for sequence_batch in truthloader:
            count += 1
            print(count)
            sequence_batch = sequence_batch.to(model.device)
            batch_loss = 0
            masked_batch, ignore_tensor = masking(sequence_batch, 0.15, model.tokens['[MASK]'], model.batch_size)
            out = model(masked_batch)
            batch_loss = model.loss_function(torch.transpose(out, 1, 2), ignore_tensor)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Apply gradient clipping
            model.optimizer.step()
            model.optimizer.zero_grad()
            epoch_loss += batch_loss
        model.scheduler.step()
        if epoch % model.log_interval == 0 or epoch == 0:
            loss_string = "Epoch " + str(epoch+1) +  " loss " + str(epoch_loss/len(truthloader))
            print(loss_string)
            val_loss = str(evaluate(model, epoch + 1))
            print("Validation Loss", val_loss)

test_model = LSTMModel(64, 'one.fasta', 'one_test.fasta', 'aa', 512)
train(test_model)






