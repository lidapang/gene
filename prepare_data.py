#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd

from logging.config import fileConfig
from pprint import pprint

class PrepareData():
    def __init__(self,
                 genotype_path="./data/genotype.dat",
                 phenotype_path="./data/phenotype.txt",
                 gene_dir="./data/gene_info/",
                 multi_pheno_path="./data/multi_phenos.txt"):
        fileConfig("logging_config.ini")
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__genotype_path = genotype_path
        self.__logger.info("Genotype file name: %s", self.__genotype_path)
        self.__phenotype_path = phenotype_path
        self.__logger.info("Phenotype file name: %s", self.__phenotype_path)
        self.__gene_info_dir = gene_dir
        self.__logger.info("Gene info dir name: %s", self.__gene_info_dir)
        self.__multi_pheno_path = multi_pheno_path
        self.__logger.info("Multigeno info file path: %s", self.__multi_pheno_path)

    def __parse_line(self, line):
        raw = line.split(" ")
        raw[-1] = raw[-1][0:-1]
        return raw

    @property
    def raw_data(self):
        self.__logger.info("Starting handling raw data file: %s", self.__genotype_path)
        data = []
        fix_error = {'II': 'TT', 'ID': 'TC', 'DD': 'CC'}

        transform = {
            'AA': 2,
            'CC': 4,
            'TT': 6,
            'GG': 12,
            'AC': 3,
            'CA': 3,
            'AT': 4,
            'TA': 4,
            'AG': 7,
            'GA': 7,
            'CT': 5,
            'TC': 5,
            'CG': 8,
            'GC': 8,
            'GT': 9,
            'TG': 9,
        }

        with open(self.__genotype_path) as data_file:
            for line in data_file.readlines():
                tmp_line = self.__parse_line(line)
                fix_error_line = [fix_error[x] if x in fix_error else x
                                  for x in tmp_line]
                transform_line = [transform[x] if x in transform else x
                                  for x in fix_error_line]
                data.append(transform_line)
        return data

    @property
    def tag(self):
        self.__logger.info("Staring handling tags file: %s", self.__phenotype_path)
        tags = []

        with open(self.__phenotype_path) as tag_file:
            for line in tag_file.readlines():
                tags.append(line[0])
        return tags

    def __read_gene_content(self, gene_num):
        file_name = self.__gene_info_dir + "gene_" + str(gene_num) + ".dat"
        file_content = []

        with open(file_name) as file_data:
            for line in file_data.readlines():
                file_content.append(line[:-1])

        return file_content

    def __process_one_gene(self, gene_num):
        importances = self.__importances
        gene_data = self.__read_gene_content(gene_num)
        rate = 0

        for importance in importances:
            if importance in gene_data:
                rate = rate + importances[importance]

        return rate

    def process_all_gene(self):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        index_data = self.raw_data[0]

        forest = RandomForestClassifier(n_estimators=8000, random_state=0, n_jobs=-1, oob_score=True)
        forest.fit(self.raw_data[1:], self.tag)

        importances = forest.feature_importances_

        indices = np.argsort(importances)[::-1]
        self.__importances = dict(map(lambda x: (index_data[x], importances[x]), indices[0:2]))

        all_file_index = list(range(1, 301))
        result = list(map(lambda x: (x, self.__process_one_gene(x)), all_file_index))
        print(forest.oob_score_)
        pprint(sorted(result, key = lambda x: x[1]))

    @property
    def multi_pheno_data(self):
        data = []
        fix_format = {'0': 0, '1': 1}

        with open(self.__multi_pheno_path) as file_data:
            for line in file_data.readlines():
                tmp_line = self.__parse_line(line)
                fix_format_line = [fix_format[x] if x in fix_format else x for x in tmp_line]
                data.append(fix_format_line)

        return data

    @property
    def multi_pheno_tags(self):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=2, n_jobs=-1, random_state=150)
        kmeans.fit(self.multi_pheno_data)

        return kmeans.labels_

if __name__ == "__main__":

    from sklearn.ensemble import RandomForestClassifier

    prepared_data = PrepareData()
    labels = prepared_data.tag
    raw_data = prepared_data.raw_data
    training_data = raw_data[1:]
    index_data = raw_data[0]

    forest = RandomForestClassifier(n_estimators=3000, random_state=0, n_jobs=-1, oob_score=True)
    forest.fit(training_data, labels)

    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    tmp = dict(map(lambda x: (index_data[x], importances[x]), indices[0:20]))
    print(tmp)

    print(importances)
    print(indices)

    print(forest.oob_score_)
    data2save = []

    for f in range(len(index_data)):
        print("%2d) %-*s %f" % (indices[f], 30, index_data[indices[f]], importances[indices[f]]))
        data2save.append([indices[f], index_data[indices[f]], importances[indices[f]]])

    result_df_2save = pd.DataFrame(data=data2save, columns=["index", "name", "importance"])
    result_df_2save.to_csv("./result/2.csv")

