import numpy as np
import pandas
import json

def tab_to_npy(tab_file):

    tab = pandas.read_table(tab_file, header=0, index_col=0)

    tab.pop("Unnamed: 242")

    with open('temp_csn_headings.json', 'wb') as outfile:
        json.dump(list(tab.columns), outfile)

    dist = tab.values

    np.save('temp_csn_identities.npy', dist)


if __name__ == '__main__':

    tab_to_npy('./clique_jac_n700.tab')
