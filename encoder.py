import numpy as np
#import pytorch

chars_aa = 'ARNDCQEGHILKMFPSTWYVX'
token_index_aa = {amino_acid:index for (index, amino_acid) in enumerate(chars_aa)}

for synonym_aa in 'OUBZ':
    token_index_aa[synonym_aa] = len(chars_aa)

extra_tokens = ['[START]', '[END]', '[PAD]']
for i in range(len(extra_tokens)):
    token_index_aa[extra_tokens[i]] = len(chars_aa) + 1 + i

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
    token_index_codon[extra_tokens[i]] = len(chars_codon) + 1 + i

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
