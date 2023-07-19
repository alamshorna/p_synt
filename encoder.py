import numpy as np
import torch
from Bio import SeqIO
import random

#chars_aa = 'ARNDCQEGHILKMFPSTWYVX'
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
    #print([token_index_codon['[START]']] + [token_index_codon[codon_mapping[codon]] for codon in codon_list] + [token_index_codon['[END]']])
    return [token_index_codon['[START]']] + [token_index_codon[codon_mapping[codon]] for codon in codon_list] + [token_index_codon['[END]']]


some_sentence = "ATGTCTACAAACTCGATAAAATTACTCGCCAGCGATGTGCATAGAGGACTCGCTGAATTAGTTGCGAGGAGGCTAGGTTTGCACATACTACCATGTGAGTTGAAAAGGGAATCCACGGGGGAAGTTCAATTCTCTATTGGGGAATCAGTTAGAGACGAAGATGTTTTTATTGTTTGTCAGATTGGTTCTGGCGAGGTAAATGACAGGGTGATTGAGCTCATGATCATGATTAACGCTTGTAAAACAGCTAGTGCTAGAAGAATCACCGTTATATTGCCAAACTTTCCTTACGCAAGACAAGACCGAAAAGATAAGTCGCGTGCTCCCATCACTGCGAAGCTAATGGCCGACATGTTGACGACTGCTGGGTGCGACCATGTTATCACCATGGATTTGCACGCTTCTCAGATTCAAGGATTCTTTGATGTCCCAGTGGATAATTTGTATGCCGAGCCTAGTGTTGTTAGGTATATAAAGGAGAAAATAGATTACAAGAACGCAATAATCATTTCGCCGGATGCTGGTGGTGCCAAGAGAGCTGCAGGGCTCGCAGACAGGCTCGACTTGAACTTTGCATTGATTCACAAAGAGCGTGCAAAGGCAAACGAAGTCTCTAGAATGGTGTTGGTGGGTGACGTGAGCGATAAAGTTTGTGTTATTGTTGACGATATGGCAGACACATGTGGTACCTTGGCGAAAGCTGCAGAGGTTTTATTGGAGAACAATGCGAAAGAAGTGATTGCCATTGTAACACATGGTATTTTGTCTGGTAATGCCATGAAGAATATCAATAACTCTAAACTTGAGAGGGTCGTATGTACAAATACGGTTCCTTTTGAGGATAAGTTGAAGTTGTGCAACAAGTTGGATACCATTGATGTTTCAGCTGTTATTGCCGAGGCTATAAGGAGATTGCACAATGGTGAGAGTATCTCTTATTTGTTCAAAAATGCACCTTTATAA"

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
    #print(data)
    return data
    #print(count)
    #print(fully_masked)
    return data

# def batching(batch_size, data, alphabet, do_mask):
#     """
#     creates a tensor of size batches of tokens 
#     """
#     #remove concatenation
#     #concatenated_data = torch.cat(data)

#     #call to the masking function
#     if do_mask == True:
#         concatenated_data = masking(concatenated_data, 0.5, alphabet)
#     num_full_batches = len(concatenated_data)//batch_size
#     extra = len(concatenated_data)%batch_size
#     amt_padding = batch_size-extra
#     pad_token = token_index_aa["[PAD]"] if alphabet == 'aa' else token_index_codon["[PAD]"]
#     padding_list = [pad_token] * amt_padding
#     padding_tensor = torch.tensor(padding_list, dtype = torch.long)
#     concatenated_data = torch.cat((concatenated_data, padding_tensor))
#     #data = concatenated_data.view(batch_size, num_full_batches+1)
#     return concatenated_data


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
