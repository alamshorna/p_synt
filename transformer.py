# TODO: remove any unnecessary imports
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

class AATokenizer:
    def __init__(self):
        self.start = '[START]'
        self.end = '[END]' 
        self.pad = '[PAD]' 
        self.mask = '[MASK]' 

        aa_chars = 'CSTAGPDEQNHRKMILVWYFX'
        aa_array = np.array([*[aa_char for aa_char in aa_chars], *[self.start, self.end, self.pad, self.mask]])

        self.tok_dict = {word: index for index, word in enumerate(aa_array)} 
        # encode synonyms
        self.tok_dict['O'], self.tok_dict['U'], self.tok_dict['B'], self.tok_dict['Z'] = 11, 4, 20, 20

        self.vocab = [key for key in self.tok_dict.keys()] # unclear if this is necessary?
        self.vocab_size = len(self.vocab)

    def encode(self, aa_sequence: str) -> torch.Tensor:
        token_tensor = [self.start] + [aa for aa in aa_sequence] + [self.end]
        id_tensor = torch.tensor([self.tok_dict[token] for token in token_tensor])
        return id_tensor

    def decode(self, token_tensor: torch.Tensor) -> str:
        decoded_str = []
        for token in token_tensor:
          decoded_str.append(self.vocab[token])
        return ' '.join(decoded_str)
    
    def pad_seq(self, aa_tok_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(aa_tok_list, batch_first=True, padding_value=self.tok_dict[self.pad])

class AADataset(Dataset):
    def __init__(self, tokenizer: AATokenizer, sequences: List[str], max_tokens: int):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.sequences)
  
    def __getitem__(self, index):
        # TODO: rewrite this to choose a random window
        return self.tokenizer.encode(self.sequences[index])[:self.max_tokens]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super(PositionalEncoding, self).__init__()
        # TODO: figure out need to cast this to int?
        pe = torch.zeros(int(max_len), d_model)
        print(pe.size())
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Transformer):
    # adapted from Pytorch's built-in transformer
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp)

        self.input_emb = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken) #replace this with a pytorch decoder layer?

        self.init_weights()

    def generate_mask(self, dim):
        # NOTE: this mask only permits attending to tokens prior to the current token
        return torch.tril(torch.ones(dim, dim))
        # NOTE: this mask permits attending to all tokens in the sequence during transformer model training
        return torch.log((torch.ones(dim,dim)))

    def init_weights(self):
        initrange = 0.1
        # TODO: look into these, and determine if this seems like the best possible choice - xavier initialization?
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self.generate_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return output

    def predict_next_token(self, input_ids):
        with torch.no_grad():
            out = self.forward(input_ids)
            new_token = torch.argmax(out[:, [-1]], -1)
            # TODO: cleaner way to do this?
            # distribution = F.softmax(out[:, -1], dim=-1).squeeze(0) #.unsqueeze(0)
            distribution = out[:, -1].squeeze(0)
            input_ids = torch.cat([input_ids, new_token], dim=1)
        return input_ids, distribution
    
class DialogueLoss(nn.Module):
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
    fasta_file = 'data/19.9K_proteins.fasta'
    test_fasta_file = 'data/mini_test_aa.fasta'

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

    
    model = TransformerModel(aa_tokenizer.vocab_size, ninp=64, nhead=8, nhid=2, nlayers=4)
    NUM_EPOCHS = 20

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = DialogueLoss()
    

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

        # scheduler.step()
        inp = torch.tensor([21]).unsqueeze(0)

        # print(inp)
        # print(model.generate(inp, 10))
        # print(aa_tokenizer.decode(model.generate(inp, 10)[0]))
        print(f"epoch {epoch}: loss = {sum(losses)/len(losses)}")
    
    # evaluation loop

    aa_test_fasta_iterator = iter(list(SeqIO.parse(open(test_fasta_file), 'fasta')))
    aa_test_sequences = [str(record.seq) for record in aa_test_fasta_iterator]
    aa_test_dataset = AADataset(aa_tokenizer, aa_test_sequences, max_tokens=512)
    aa_test_dataloader = torch.utils.data.DataLoader(aa_test_dataset, batch_size=32, collate_fn=aa_collate)

    
    preds = {aa_token: torch.tensor([0]*29).float() for aa_token in range(len(aa_tokenizer.vocab))}
    for seq_dict in tqdm.tqdm(aa_test_dataloader):
        for i in range(seq_dict["seq"].size(0) - 1):  
            for j in range(1, seq_dict["seq"].size(1) - 1):
                context = seq_dict["seq"][i][:j].unsqueeze(0)  
                pred_dist = model.predict_next_token(context)[1]
                masked_token = seq_dict["seq"][i][j].item()
                if masked_token != 23:
                    preds[masked_token] = torch.add(preds[masked_token], pred_dist).float()
    preds = np.array([np.array(F.log_softmax(value, dim=-1)) for key, value in preds.items()])
    sns.heatmap(preds[:20, :20])
    path = 'heatmap' + '.png'
    plt.savefig(path)

if __name__=="__main__":
    main()