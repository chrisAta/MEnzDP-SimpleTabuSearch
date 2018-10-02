import max_min_diversity


if __name__ == '__main__':

    for i in range(5, 245, 5):

        ssn_set = max_min_diversity.compute_diverse_set('./temp_ssn_identities.npy',
                                                'temp_ssn_headings.json', i)


        csn_set = max_min_diversity.compute_diverse_set('./temp_csn_identities.npy',
                                                'temp_csn_headings.json', i)

        comb_set = set(ssn_set + csn_set)

        temp_set = []

        # print ssn_set
        # print csn_set

        temp_set = [x for x in csn_set if x not in ssn_set]

        temp_set += [x for x in ssn_set if x not in csn_set]

        val = (len(temp_set))
        # val = (i * 2.0 - len(comb_set)) / i
        # val = (i - (len(comb_set) - float(i))) / i

        # print i
        # print len(comb_set) - i
        print val

        # print len(temp_set)
        # print sorted(temp_set)
        # print sorted(comb_set)
        # print len(comb_set)
        # max_min_diversity.compute_diverse_set('./temp_csn_identities.npy',
        #                                         'temp_csn_headings.json', 60, True)
