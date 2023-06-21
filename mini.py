import pandas as pd
import numpy as np
from sklearn import model_selection as skms
from Bio import SeqIO

fasta_data = list(SeqIO.parse(open('data/10K_codons_test.fasta'), 'fasta'))

mini_dataset = fasta_data[0:1500]
with open('data/mini_codon.fasta', 'w') as data_file:
    for datapoint in mini_dataset:
        data_file.write('>' + str(datapoint.id)+ '\n')
        data_file.write(str(datapoint.seq) + '\n')
        
mini_testset = fasta_data[1501:2000]
with open('data/mini_test_codon.fasta', 'w') as test_file:
    for testpoint in mini_testset:
        test_file.write('>' + str(testpoint.id)+ '\n')
        test_file.write(str(testpoint.seq) + '\n')