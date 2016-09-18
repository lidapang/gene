import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prepare_data import PrepareData
from pprint import pprint

if __name__ == "__main__":
    p = PrepareData()
    result = p.process_all_gene()
    result = sorted(result, key=lambda x: x[1], reverse=True)

    index = [x[0] for x in result]
    value = [x[1] for x in result]

    # print([index, value])
    df = pd.DataFrame(data=result, columns=["index", "value"], )
    df.to_csv("./result/3.csv", index=False)

    ## Plot
    plt.figure()
    plt.bar(index, value, width=.2, color='g')
    plt.savefig('./result/3.png')
    plt.close()

