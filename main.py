import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

phenotype_path = "./data/genotype.dat"

def parse_line(line):
    raw = line.split(" ")
    raw[-1] = raw[-1][0:-1]
    return raw

def parse_file(file_path):
    data = []
    fix_error = {'II': 'TT', 'ID': 'TC'}
    transform = {'CA': 0, 'CC': 1, 'TC': 2, 'AT': 3, 'CG': 4, 'AC': 5, 'TA': 6, 'AA': 7, 'TT': 8, 'AG': 9, 'GT': 10, 'GG': 11, 'TG': 12, 'GC': 13, 'CT': 14, 'GA': 15}

    with open(file_path) as data_file:
        for line in data_file.readlines():
            tmp_line = parse_line(line)
            fix_error_line = [fix_error[x] if x in fix_error else x for x in tmp_line]
            transform_line = [transform[x] if x in transform else x for x in fix_error_line]
            data.append(transform_line)

    return data

raw_data = parse_file(phenotype_path)
data_frame = pd.DataFrame(raw_data[1:], columns=raw_data[0])
# fac = pd.factorize(data_frame['rs3094315'])
print(data_frame)
# print(fac[0])
# print(type(fac[0]))
# print(len(fac[0]))
