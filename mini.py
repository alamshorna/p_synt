import pandas as pd
import numpy as np
from sklearn import model_selection as skms
from Bio import SeqIO

fasta_codon_data = list(SeqIO.parse(open('data/10K_codons_test.fasta'), 'fasta'))

mini_dataset = fasta_codon_data[0:20]

with open('data/mini.fasta', 'w') as data_file:
    for datapoint in mini_dataset:
        data_file.write('>' + str(datapoint['Name'] + '\n'))
        data_file.write(str(datapoint['seq']))
        print(datapoint)
        


            