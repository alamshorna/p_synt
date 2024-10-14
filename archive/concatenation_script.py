import pandas as pd
import numpy as np
from sklearn import model_selection as skms
from Bio import SeqIO

fasta_prot_data = list(SeqIO.parse(open('zebrafish/zebrafish.1.protein.faa'), 'fasta'))
fasta_dna_data = list(SeqIO.parse(open('zebrafish/zebrafish.1.rna-2.fna'), 'fasta'))

prot_data_dict, dna_data_dict = {}, {}
shared_dict = {}
num_prot_seq, num_dna_seq = 0,0
prot_ids, dna_ids = set(), set()

def translation(dna_seq, protein_seq):
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
    dna_seq, protein_seq = str(dna_seq), str(protein_seq)
    
    if dna_seq.find('AAAAAAAAA')!= -1:
        final_index = dna_seq.find('AAAAAAAAA') - (dna_seq.find('AAAAAAAAA')%3) + 1
    else:
        final_index = len(dna_seq)
    
    
    dna_seq = dna_seq[dna_seq.find('ATG'): final_index]

    # print(dna_seq)
    # print('')
    # print(protein_seq)
    generate_protein = ''
    for i in range(0, len(dna_seq), 3):
        codon = dna_seq[i:i+3]
        if (codon in amino.keys()) and (amino[codon] == '$' or len(codon)!=3):
            break
        # print(codon)
        if codon in amino.keys():
            generate_protein = generate_protein + amino[codon]
        else:
            return None
    return generate_protein


for fasta in fasta_prot_data:
    prot_data_dict[fasta.id] = fasta.seq
    shared_dict[fasta.id[3:]] = [fasta.id, None] 
    prot_ids.add(fasta.id[3:])
    num_prot_seq += 1

for fasta in fasta_dna_data:
    dna_data_dict[fasta.id] = fasta.seq
    if fasta.id[3:] in shared_dict.keys():
        shared_dict[fasta.id[3:]][1] = fasta.id
    else:
        shared_dict[fasta.id[3:]] = [None, fasta.id]
    dna_ids.add(fasta.id[3:])
    num_dna_seq += 1

shared_ids = list(dna_ids.intersection(prot_ids))
#print(shared_ids)
print(shared_dict[shared_ids[0]])
count = 0
with open("zebrafish\zebrafish.1.shared_protein.fasta", 'w') as prot_file:
    with open("zebrafish\zebrafish.1.shared_dna.fasta", 'w') as dna_file:
        for id in shared_ids:
            dna_name = shared_dict[id][1]
            dna_sequence = dna_data_dict[dna_name]
            # dna_file.write('>' + str(dna_name) + '\n')
            # dna_file.write(str(dna_sequence) + '\n')
            # dna_file.write('\n')
            
            prot_name = shared_dict[id][0]
            prot_sequence = prot_data_dict[prot_name]
            # prot_file.write('>' + str(prot_name) + '\n')
            # prot_file.write(str(prot_sequence) + '\n')
            # prot_file.write('\n')

            # print(translation(dna_sequence, prot_sequence))
            # print(dna_sequence)
            # print(prot_sequence)
            pseudoprotein = translation(dna_sequence, prot_sequence)
        
            if pseudoprotein!=None:
                if pseudoprotein == prot_sequence:
                    print(pseudoprotein)
                    count += 1
                # print('pseudoprotein', pseudoprotein)
                # print('actual', prot_sequence)
print(count)

            



# for id in shared_ids:
#     print(shared_dict[id])



# print('first', id_differences[0])
# print(len(id_differences))
# print(len(prot_data_dict))
# print(len(dna_data_dict))


# for protein in fasta_prot_data:
#     SeqIO.write(output, protein, 'fasta')


