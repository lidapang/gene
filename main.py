import Aprio as AP
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

phenotype_path = "./data/genotype.dat"
# phenotype_path = "D:\建模\gene\data\genotype.dat"

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
# write_path = "./data/question1_0.dat"
# try:
#     with open(write_path, 'w') as write_path:
#         for line in raw_data:
#             write_path.writelines(str(line)+"\n")
# except FileNotFoundError:
#     f = open(write_path)
#     for line in data_frame:
#         f.writelines(line)
#
#     f.close()

# data_lines = pd.read_table('./data/question1_1_noflag.txt', header=None, encoding='utf8', delim_whitespace=True,
#                            index_col=0)
# print(data_lines)

# a = AP.Apriori(data_frame)

# print(a.do())

noflag_path = "./data/question1_1_noflag.txt"
with open(noflag_path) as noflag:
    data_noflag = []
    dir_noflag = {}
    for line in noflag:
        data_noflag.append(line)
        dir_noflag.s

print(type(data_noflag))
print(len(data_noflag))
