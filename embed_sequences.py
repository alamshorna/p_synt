# from __future__ import print_function,division

# import sys
# import numpy as np
# import h5py

# import torch

# from prose.alphabets import Uniprot21, NucleotideCodons
# import prose.fasta as fasta
# from types import SimpleNamespace
# from tqdm import tqdm
# from collections import defaultdict
# import pandas as pd
# from train_prose_masked_2 import NucleotideModule
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import pickle

# def embed_sequence(model, x, alphabet, pool='none', use_cuda=False):
#     if len(x) == 0:
#         n = model.embedding.proj.weight.size(1)
#         z = np.zeros((1, n), dtype=np.float32)
#         return z

#     # Convert to alphabet index
#     x = alphabet.encode(x)
#     x = torch.from_numpy(x)
#     if use_cuda:
#         x = x.cuda()

#     # Embed the sequence
#     with torch.no_grad():
#         x = x.long().unsqueeze(0)
#         z = model.transform(x)
#     return z


# def get_logits(model, x, alphabet, pool='none', use_cuda=False):
#     if len(x) == 0:
#         n = model.embedding.proj.weight.size(1)
#         z = np.zeros((1, n), dtype=np.float32)
#         return z

#     # Convert to alphabet index
#     x = alphabet.encode(x)
#     x = torch.from_numpy(x)
#     if use_cuda:
#         x = x.cuda()

#     # Return logit
#     with torch.no_grad():
#         x = x.long().unsqueeze(0)
#         z = model.forward(x)
#         # print('logits shape: ', torch.logit(z).cpu().numpy().shape)
#         # print('hidden_shape:', z.last_hidden_state.shape)
#         z = z.cpu().numpy()
#         return z


# def main(alphabet, args):
#     """
#     import argparse
#     parser = argparse.ArgumentParser()

#     parser.add_argument('path')
#     parser.add_argument('-m', '--model', default='prose_mt', help='pretrained model to load, prose_mt loads the pretrained ProSE MT model, prose_dlm loads the pretrained Prose DLM model, otherwise unpickles torch model directly (default: prose_mt)')
#     parser.add_argument('-o', '--output')
#     parser.add_argument('--pool', choices=['none', 'sum', 'max', 'avg'], default='none', help='apply some sort of pooling operation over each sequence (default: none)')
#     parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')

#     args = parser.parse_args()
#     print(args)
#     """

#     args = SimpleNamespace(**args)

#     args2 = {
#         # "path_train": 'data/zebrafish/zebrafish_p1_aa_train.fa', #X.fa',
#         # "path_val": "data/zebrafish/zebrafish_p1_aa_train.fa", #X.fa",
#         "path_train": 'data/zebrafish/zebrafish_p1_dna_train.fa', #X.fa',
#         "path_val": "data/zebrafish/zebrafish_p1_dna_train.fa", #X.fa",
#         "model": None,
#         "resume": None,
#         "rnn_dim": 512,
#         "num_layers": 3,
#         "dropout": 0.0,
#         "num_steps": 10, #1, #100, #1000000,
#         "save_interval": 500,
#         "max_length": 1024,
#         "p": 0.1,
#         "batch_size": 20,
#         "weight_decay": 0,
#         "lr": 0.0005,
#         "save_prefix": 'data/result/',
#         "device": [0, 1, 2, 3],
#         "debug": False,
#         "wandb_id": None,
#         "wandb_project": "nucleotide",
#         "wandb_entity": "mreveiz",
#         "wandb": False,
#         "experiment_name": "First",
#         "alphabet": alphabet,
#         "num_workers": 4,
#         "epoch_len": 1024,
#         "max_epochs": 2 
#         #600
#     }

#     nin = len(args2["alphabet"])
#     nout = len(args2["alphabet"])
#     hidden_dim = args2["rnn_dim"]

#     path = args.path

#     # load the model
#     if args.model == 'prose_mt':
#         from prose.models.multitask import ProSEMT
#         print('# Loading the pre-trained ProSE MT model', file=sys.stderr)
#         model = ProSEMT.load_pretrained()
#     elif args.model == 'prose_dlm':
#         from prose.models.lstm import SkipLSTM
#         print('# Loading the pre-trained ProSE DLM model', file=sys.stderr)
#         model = SkipLSTM.load_pretrained()
#     else:
#         # print('# Loading model:', args.model, file=sys.stderr)
#         model = NucleotideModule(
#             lr=args2["lr"],
#             nin=nin,
#             nout=nout,
#             hidden_dim=hidden_dim,
#             num_layers=args2["num_layers"],
#             dropout=args2["dropout"],
#             weight_decay=args2["weight_decay"],
#             alphabet=args2["alphabet"],
#             batch_size=args2["batch_size"],
#         )
        
#         state_dict = torch.load(args.model, map_location = torch.device('cpu'))
        
#         from collections import OrderedDict
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#                 name = 'model.' + str(k)
#                 new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)

#         # model = torch.load(args.model)
#                         #    map_location = torch.device('cpu'))
#         #    map_location="cuda:{}".format(args.device))
#         #  _use_new_zipfile_serialization=False
#     model.eval()

#     # set the device
#     d = args.device
#     use_cuda = (d != -1) and torch.cuda.is_available()
#     if d >= 0:
#         torch.cuda.set_device(d)

#     if use_cuda:
#         model = model.cuda()

#     # parse the sequences and embed them
#     # write them to hdf5 file
#     print('# Writing:', args.output, file=sys.stderr)

#     pool = args.pool
#     print('# Embedding with pool={}'.format(pool), file=sys.stderr)

#     with open(path) as f:
#         max_count = sum(1 for _ in f) // 2

#     count = 0
#     # all_logits = -np.ones((max_count, args.max_length, len(alphabet)))
#     with open(path, 'rb') as f:
#         for name, sequence in tqdm(fasta.parse_stream(f), total=max_count):
#             # z = get_logits(model, sequence, alphabet, pool=pool, use_cuda=use_cuda)
#             # all_logits[count, 0:z.shape[1], 0:z.shape[2]] = z[0, 0:args.max_length, :].copy()

#             embed = embed_sequence(model, sequence, alphabet, pool=pool, use_cuda=use_cuda)
            
#             unpacked_seq, _ = pad_packed_sequence(embed, batch_first=True)
#             # unpacked_seq_torch = unpacked_seq[0,:,:].cpu()
#             # unpacked_seq_np = unpacked_seq[0,:,:].cpu().numpy()
#             unpacked_seq_np = unpacked_seq.cpu().numpy()
            
#             # np.savetxt(args.output_embed, unpacked_seq_np, delimiter=', ')

#             # with open(args.output_embed_pkl, 'wb') as handle:
#             #     pickle.dump(unpacked_seq_torch, handle)
#             filepath = str(args.output_embed_pkl) + '_' + str(count) + '.pkl'
#             # np.save(filepath, unpacked_seq_np)

#             with open(filepath, 'wb') as handle:
#                 pickle.dump(unpacked_seq_np, handle)

#             # print('Unpack:', unpacked_seq)
#             # print('length:', unpacked_lens)
#             # print(unpacked_seq.shape)
#             # print(embed.cpu())
#             # .numpy()[:,-1])
#             # np.savetxt(args.output_embed, embed.cpu().numpy(), delimiter=', ')
#             # with open(args.output_embed, 'w') as file:
#             #     file.write(embed.cpu().numpy())

#             count += 1
#     print(' '*80, file=sys.stderr, end='\r')
#     # np.save(args.output, z)
#     # buffer = io.BytesIO()


# if __name__ == '__main__':
#     alphabet = NucleotideCodons()              # for dna_codon files --> train.py
#     # alphabet = Uniprot21()              # for aa_prot files --> train2.py
#     """
#     parser.add_argument('path')
#     parser.add_argument('-m', '--model', default='prose_mt',
#                         help='pretrained model to load, prose_mt loads the pretrained ProSE MT model, prose_dlm loads the pretrained Prose DLM model, otherwise unpickles torch model directly (default: prose_mt)')
#     parser.add_argument('-o', '--output')
#     parser.add_argument('--pool', choices=['none', 'sum', 'max', 'avg'], default='none',
#                         help='apply some sort of pooling operation over each sequence (default: none)')
#     parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')
    
#     """

#     # args = {"path": "data/zebrafish/zebrafish_p1_aa_test.fa", "model": "/net/scratch3.mit.edu/scratch3-3/iarora/ishadat_store/results/saved_model/train2_aa_prot_alphabet_iter_4-14-23_state_dict_300_epochs.pth", "output": "data/result/logits/logits_aa_alphabet_4-14-23__300_epochs",
#     #         "pool": None, "device": 0, "max_length": 1024, "output_embed": "data/result/logits/embed_aa_alphabet_4-14-23__300_epochs.txt",
#     #         "output_embed_pkl": "data/result/logits/aa_epoch_300/embed_aa_alphabet_4-14-23__300_epochs"}

#     args = {"path": "data/zebrafish/zebrafish_p1_dna_test.fa", "model": "data/result/saved_model/train_dna_codon_alphabet_iter_4-14-23_state_dict_300_epochs.pth", "output": "data/result/logits/logits_codons_alphabet_4-14-23__300_epochs",
#             "pool": None, "device": 0, "max_length": 1024, "output_embed": "data/result/logits/embed_codons_alphabet_4-14-23__300_epochs",
#             "output_embed_pkl": "data/result/logits/codons_epoch_300/embed_codons_alphabet_4-14-23__300_epochs"}

#     # /net/scratch3.mit.edu/scratch3-3/iarora/ishadat_store/results/

#     main(alphabet, args)