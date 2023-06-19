import numpy as np
import torch
from Bio import SeqIO
import random

chars_aa = 'ARNDCQEGHILKMFPSTWYVX'
token_index_aa = {amino_acid:index for (index, amino_acid) in enumerate(chars_aa)}

#rewrite this
for synonym_aa in 'OUBZ':
    token_index_aa[synonym_aa] = len(chars_aa)

#added a "[PAD]" token separately in the token dictionary (as in proteinBERT)
#alternative option: set to 0, as with the nn.embedding() function
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
    return [token_index_codon['[START]']] + [token_index_codon[amino[codon]] for codon in codon_list] + [token_index_codon['[END]']]
    tokens = [token_index_codon['[START]']] + [token_index_codon[amino[codon]] for codon in codon_list] + [token_index_codon['[END]']]
    return (tokens, len(tokens))


some_sentence = "ATGTCTACAAACTCGATAAAATTACTCGCCAGCGATGTGCATAGAGGACTCGCTGAATTAGTTGCGAGGAGGCTAGGTTTGCACATACTACCATGTGAGTTGAAAAGGGAATCCACGGGGGAAGTTCAATTCTCTATTGGGGAATCAGTTAGAGACGAAGATGTTTTTATTGTTTGTCAGATTGGTTCTGGCGAGGTAAATGACAGGGTGATTGAGCTCATGATCATGATTAACGCTTGTAAAACAGCTAGTGCTAGAAGAATCACCGTTATATTGCCAAACTTTCCTTACGCAAGACAAGACCGAAAAGATAAGTCGCGTGCTCCCATCACTGCGAAGCTAATGGCCGACATGTTGACGACTGCTGGGTGCGACCATGTTATCACCATGGATTTGCACGCTTCTCAGATTCAAGGATTCTTTGATGTCCCAGTGGATAATTTGTATGCCGAGCCTAGTGTTGTTAGGTATATAAAGGAGAAAATAGATTACAAGAACGCAATAATCATTTCGCCGGATGCTGGTGGTGCCAAGAGAGCTGCAGGGCTCGCAGACAGGCTCGACTTGAACTTTGCATTGATTCACAAAGAGCGTGCAAAGGCAAACGAAGTCTCTAGAATGGTGTTGGTGGGTGACGTGAGCGATAAAGTTTGTGTTATTGTTGACGATATGGCAGACACATGTGGTACCTTGGCGAAAGCTGCAGAGGTTTTATTGGAGAACAATGCGAAAGAAGTGATTGCCATTGTAACACATGGTATTTTGTCTGGTAATGCCATGAAGAATATCAATAACTCTAAACTTGAGAGGGTCGTATGTACAAATACGGTTCCTTTTGAGGATAAGTTGAAGTTGTGCAACAAGTTGGATACCATTGATGTTTCAGCTGTTATTGCCGAGGCTATAAGGAGATTGCACAATGGTGAGAGTATCTCTTATTTGTTCAAAAATGCACCTTTATAA"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#masking function
def masking(data, masking_proportion, alphabet):
    """
    randomly chooses masking_proportion of tokens to cover
    returns data with indices masked
    """
    #will mask start and end tokens freely among the chosen 15%
    amt_data = len(data)
    num_desired = int(masking_proportion*amt_data)
    to_be_masked = torch.randint(amt_data-1, (num_desired,))
    chosen_indices = [random.sample(list(range(amt_data)), num_desired)]
    masking_choices = [random.choices(['null', 'correct', 'incorrect'], [.8, .1, .1], k=num_desired)]
    alphabet_dictionary = token_index_aa if alphabet == 'aa' else token_index_codon
    for i in range(len(chosen_indices)):
        if masking_choices[i] == 'null':
            data[chosen_indices[i]] = '[MASK]'
        elif masking_choices[i] == 'correct':
            pass
        elif masking_choices[i] == 'incorrect':
            #rewrite to not include the correct amino acid
            data[chosen_indices[i]] = alphabet_dictionary[random.choice(list(range(0, len(alphabet_dictionary)-1)))]
    return data
    

def iterate(length):
    """
    given a number
    returns a list of length length
    """
    return  list(range(length))

def batching(batch_size, data, alphabet, do_mask):
    """
    creates a tensor of size batches of tokens 
    """
    concatenated_data = torch.cat(data)
    #call to the masking function
    if do_mask == True:
        concatenated_data = masking(concatenated_data, 0.15, alphabet)
    num_full_batches = len(concatenated_data)//batch_size
    extra = len(concatenated_data)%batch_size
    amt_padding = batch_size-extra
    pad_token = token_index_aa["[PAD]"] if alphabet == 'aa' else token_index_codon["[PAD]"]
    padding_list = [pad_token] * amt_padding
    padding_tensor = torch.tensor(padding_list, dtype = torch.long)
    concatenated_data = torch.cat((concatenated_data, padding_tensor))
    #data = concatenated_data.view(batch_size, num_full_batches+1)
    return concatenated_data

def data_process(fasta_file, batch_size, alphabet, do_mask):
    """
    Performs complete data_preprocessing of amino acid sequences in a fasta_file
    including tokenization, concatenation, chunking, and padding
    input: file_path (str) containing protein sequences (and ids)
    output: set of pytorch tensors of batch_size, where the last is padded
    
    note that batch_size is inversely related to chunk_size
    """
    data_iterator = iter(list(SeqIO.parse(open(fasta_file), 'fasta')))
    #find maximum length
    tokenizer_function = encode_aa if alphabet == 'aa' else encode_codon

    #perform tokenization and convert it into a pytorch tensor
    tokenized_data = [torch.tensor(tokenizer_function(str(data.seq)), dtype = torch.long) for data in data_iterator]
    #return tokenized_data
    # off load the batching functionarlity to the pytorch dataloader
    return batching(batch_size, tokenized_data, alphabet, do_mask)

#print(data_process("/Users/shornaalam/Documents/p_synt/data/10K_codons_test.fasta", 512, 'codon'))

# #for positional_encodings:
# def indices_encoding(fasta_file, batch_size, alphabet):
#     data_iterator = iter(list(SeqIO.parse(open(fasta_file), 'fasta')))
#     tokenizer_function = encode_aa if alphabet == 'aa' else encode_codon
#     data_iterator = iter(list(SeqIO.parse(open(fasta_file), 'fasta')))
#     indexed_data = [torch.tensor(iterate(len(tokenizer_function(str(data.seq))))) for data in data_iterator]
#     return batching(batch_size, indexed_data, alphabet, True)
