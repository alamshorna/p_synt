import pandas as pd
import numpy as np
from sklearn import model_selection as skms
from Bio import SeqIO


fasta_prot_data = list(SeqIO.parse(open('20K_proteins.fasta'), 'fasta'))
fasta_test_data = list(SeqIO.parse(open('10K_proteins_test.fasta'), 'fasta'))

chars_aa = 'ARNDCQEGHILKMFPSTWYVX'
chars_codon = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789?@*'

usable_prots, usable_test_prots = [], []

for prot in fasta_prot_data:
    usable = True
    for char in prot:
        if char not in chars_aa:
            usable = False
    if usable == True:
        usable_prots.append(prot)
        #usable_prot_dict[prot.id] = prot.seq
    
for test in fasta_test_data:
    usable = True
    for char in prot:
        if char not in chars_aa:
            usable = False
    if usable == True:
        usable_test_prots.append(test)
        #usable_test_dict[test.id] = test.seq

print(len(usable_prots))
print(len(usable_test_prots))

with open('19.9K_proteins.fasta', 'w') as data_file:
    for usable_prot in usable_prots:
        data_file.write('>' + str(usable_prot.id)+ '\n')
        data_file.write(str(usable_prot.seq) + '\n')
        
