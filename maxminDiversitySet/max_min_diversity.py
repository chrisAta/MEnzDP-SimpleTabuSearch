import numpy as np
import data_prep

def greedy_min_max_alg(dist, headings, set, k, stochastic=False):

        while len(set) < k:

            min_val = 10
            min_ind = 0

            for i in range(0, len(dist)):

                if i in set:
                    continue

                rand = np.random.randint(0,10000)

                if stochastic and rand < 3:

                    print 'RANDOM ADDITION: ' + str(rand)
                    min_ind = i
                    break

                max_val = 0

                # print [dist[i,j] for j in set]

                for j in set:
                    if dist[i,j] > max_val:
                        # print dist[i,j]
                        max_val = dist[i,j]
                # print 'YAY'
                # max_val = np.nanmax(dist[i])

                if max_val < min_val:

                    min_val = max_val
                    min_ind = i
                    # print min_val

            # print "ADDED %s" % (headings[min_ind])

            set += [min_ind]

        return set



def compute_diverse_set(dist_file, heading_file, k, stochastic=False):

    set = []

    dist = data_prep.initialise_matrix(dist_file)

    headings = data_prep.initialise_headings(heading_file)

    min = data_prep.get_matrix_min(dist)

    set += [min[0], min[1]]

    # set += [np.random.randint(0,241)]
    #
    # set += [np.random.randint(0,241)]

    set = greedy_min_max_alg(dist, headings, set, k, stochastic)

    binary = []

    for i in range(0, len(headings)):
        if i in set:
            binary += [1]
        else:
            binary += [0]

    set = sorted([headings[x] for x in set])

    # print set

    return set, binary
