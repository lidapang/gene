#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from logging.config import fileConfig

class PrepareData():
    def __init__(self, genotype_path="./data/genotype.dat", phenotype_path="./data/phenotype.txt"):
        fileConfig("logging_config.ini")
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__genotype_path = genotype_path
        self.__logger.info("Genotype file name: %s", self.__genotype_path)
        self.__phenotype_path = phenotype_path
        self.__logger.info("Phenotype file name: %s", self.__phenotype_path)

    def __parse_line(self, line):
        raw = line.split(" ")
        raw[-1] = raw[-1][0:-1]
        return raw

    @property
    def raw_data(self):
        self.__logger.info("Starting handling raw data file: %s", self.__genotype_path)
        data = []
        fix_error = {'II': 'TT', 'ID': 'TC', 'DD': 'CC'}
        transform = {'CA': 0,
                     'CC': 1,
                     'TC': 2,
                     'AT': 3,
                     'CG': 4,
                     'AC': 5,
                     'TA': 6,
                     'AA': 7,
                     'TT': 8,
                     'AG': 9,
                     'GT': 10,
                     'GG': 11,
                     'TG': 12,
                     'GC': 13,
                     'CT': 14,
                     'GA': 15}

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
# raw_data = parse_file(genotype_path)
# data_frame = pd.DataFrame(raw_data[1:], columns=raw_data[0])
# fac = pd.factorize(data_frame['rs3094315'])
# print(data_frame)
# print(fac[0])
# print(type(fac[0]))
# print(len(fac[0]))
if __name__ == "__main__":
    p = PrepareData()
    print(len(p.parse_raw_data_file()[0]))
    print(p.parse_raw_data_file()[1])
    print(p.parse_tag())
