# an implementation of Google's BERT architecture, which works via masking
# goal is to imitated 2023 proteinBERT approach, with different embedding
import torch
import torch.nn as nn
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
import torch.optim as optim
import tqdm
from Bio import SeqIO
import seaborn as sns
import matplotlib.pyplot as plt
import math

from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, Tensor

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sklearn.preprocessing as skp
import umap

from aa_transformer import AADataset, AATokenizer

def pad_masking(x): # x = batch_size x seq_len
    return (x == 23) 

def causal_masking(seq_len, num_heads):
    batch_size, seq_len = seq_len.size()
    attention_mask = torch.zeros((batch_size * num_heads, seq_len, seq_len), dtype=torch.bool)
    for i in range(seq_len):
        attention_mask[:, i, i] = True  # Mask out the current token (set to True)
    return attention_mask

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, hidden_size)
        positions = torch.arange(0, max_len)
        self.register_buffer('positions', positions)

    def forward(self, sequence):
        batch_size, seq_len = sequence.size()
        positions = self.positions[:seq_len].unsqueeze(0).repeat(batch_size, 1)
        return self.positional_embedding(positions)

def build_model(layers_count, hidden_size, heads_count, d_ff, dropout_prob, max_len, vocabulary_size):
    token_embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=hidden_size)
    positional_embedding = PositionalEmbedding(max_len=max_len, hidden_size=hidden_size)
    # segment_embedding = SegmentEmbedding(hidden_size=hidden_size)

    encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=heads_count, batch_first=True)
    encoder = TransformerEncoder(encoder_layer, num_layers=layers_count)

    bert = BERT(
        encoder=encoder,
        token_embedding=token_embedding,
        positional_embedding=positional_embedding,
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size)

    return bert

class BERT(nn.Module):

    def __init__(self, encoder, token_embedding, positional_embedding, hidden_size, vocabulary_size):
        super(BERT, self).__init__()

        self.encoder = encoder
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.token_prediction_layer = nn.Linear(hidden_size, vocabulary_size)
        self.classification_layer = nn.Linear(hidden_size, 2)

    def forward(self, inputs):
        # print("INPUT", inputs.size())
        sequence = inputs
        token_embedded = self.token_embedding(sequence)
        positional_embedded = self.positional_embedding(sequence)
        embedded_sources = token_embedded + positional_embedded 
        # print("EMBEDDING", embedded_sources.size())

        pad_mask = pad_masking(sequence)
        causal_mask = causal_masking(sequence, 4)
        encoded_sources = self.encoder(embedded_sources, src_key_padding_mask=pad_mask, mask=causal_mask)
        token_predictions = self.token_prediction_layer(encoded_sources)
        # classification_embedding = encoded_sources[:, 0, :]
        # classification_output = self.classification_layer(classification_embedding)
        return token_predictions
    
class TransformerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor, inp_mask: torch.Tensor):
        logits = logits.transpose(1, 2)
        loss = self.criterion(logits[:, :, :-1], input_ids[:, 1:])
        loss = (loss[inp_mask[:, 1:] == 1]).mean()
        return loss
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: make this into a parameter
    fasta_file = 'data/identical_aa.fasta'
    test_fasta_file = 'data/identical_test_aa.fasta'

    aa_fasta_iterator = iter(list(SeqIO.parse(open(fasta_file), 'fasta')))
    aa_sequences = [str(record.seq) for record in aa_fasta_iterator]

    aa_tokenizer = AATokenizer()
    aa_dataset = AADataset(aa_tokenizer, aa_sequences, max_tokens=512)

    # reference: https://www.codefull.org/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/ 
    # for information on how to write a collate function
    def aa_collate(sequences: List[torch.Tensor]):

        lengths = torch.LongTensor([len(x) for x in sequences])
        padded_tok_seqs = aa_tokenizer.pad_seq(sequences) 
        loss_mask = (padded_tok_seqs != 23).float()
        return {"seq": padded_tok_seqs, "loss_mask": loss_mask, "length": lengths}

    aa_dataloader = torch.utils.data.DataLoader(aa_dataset, batch_size=32, collate_fn=aa_collate)

    model = build_model(layers_count=4, hidden_size=64, heads_count=4, d_ff=512, dropout_prob=0.5, max_len=512, vocabulary_size=aa_tokenizer.vocab_size)
    NUM_EPOCHS = 20

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = TransformerLoss()
    

    # training loop
    for epoch in range(NUM_EPOCHS):
        losses = []
        for seq_dict in tqdm.tqdm(aa_dataloader):
            optimizer.zero_grad()
            outputs = model(seq_dict["seq"])
            loss = criterion(outputs, seq_dict["seq"], seq_dict["loss_mask"])
            losses.append(loss)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch}: loss = {sum(losses)/len(losses)}")
    
    # evaluation loop
    aa_test_fasta_iterator = iter(list(SeqIO.parse(open(test_fasta_file), 'fasta')))
    aa_test_sequences = [str(record.seq) for record in aa_test_fasta_iterator]
    aa_test_dataset = AADataset(aa_tokenizer, aa_test_sequences, max_tokens=512)
    aa_test_dataloader = torch.utils.data.DataLoader(aa_test_dataset, batch_size=32, collate_fn=aa_collate)

    # TODO: fix magic number here :(
    preds = {aa_token: torch.tensor([0]*29).float() for aa_token in range(len(aa_tokenizer.vocab))}
    model.eval()
    for seq_dict in tqdm.tqdm(aa_test_dataloader):
        current_batch = seq_dict["seq"]
        current_pred_batch = model(seq_dict["seq"])
        print(seq_dict["seq"].size(), model(seq_dict["seq"]).size())
        # raise Exception("exception")
        for i in range(seq_dict["seq"].size(0)):  
            # TODO: parameterize magic number
            for j in range(seq_dict["seq"].size(1)):
                # print(current_batch[i].size(), current_pred_batch[i].size())
                pred_dist = current_pred_batch[i][j]
                # print("-"*40)
                # print(torch.argmax(pred_dist))
                masked_token = current_batch[i][j].item()
                # print(pred_dist, masked_token)
                # print(masked_token)
                if masked_token != 23:
                    preds[masked_token] = torch.add(preds[masked_token], pred_dist).float()
    preds = np.array([value.detach().numpy() for key, value in preds.items()])
    sns.heatmap(preds[:20, :20])
    path = 'BERT_heatmap' + '.png'
    plt.savefig(path)

if __name__=="__main__":
    main()