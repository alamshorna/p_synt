import pandas as pd
import numpy as np
from sklearn import model_selection as skms
from Bio import SeqIO

fasta_codon_data = list(SeqIO.parse(open('data/10K_proteins_test.fasta'), 'fasta'))

mini_dataset = fasta_codon_data[0:200]
print(mini_dataset)
with open('data/mini_aa.fasta', 'w') as data_file:
    for datapoint in mini_dataset:
        data_file.write('>' + str(datapoint.id)+ '\n')
        data_file.write(str(datapoint.seq) + '\n')
        print(datapoint)
        
mini_testset = fasta_codon_data[201:300]
with open('data/mini_test_aa.fasta', 'w') as test_file:
    for testpoint in mini_testset:
        test_file.write('>' + str(testpoint.id)+ '\n')
        test_file.write(str(testpoint.seq) + '\n')
        print(datapoint)