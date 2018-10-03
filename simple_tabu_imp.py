from __future__ import print_function
from tabstuSearch.TabuSearch import TabuSearch as TabuSearch
from tabstuSearch.Solution import Solution as Solution
from tabstuSearch.Path import Path as Path
from tabstuSearch.Move import Move as Move
from data_prep import initialise_headings, initialise_matrix
from maxminDiversitySet import max_min_diversity

from copy import deepcopy
from random import choice
import numpy as np


class MEnzDPTabuSearch(TabuSearch):

    def _score(self, sol):

        indices = []
        score = 0
        mat = self.opt_tuple[0]

        (set_indices, non_indices) = sep_indices(sol.val)

        for i in range(0, len(set_indices)):
            for j in range(i + 1, len(set_indices)):
                score += (1 - mat[set_indices[i], set_indices[j]])

        return score

    def _create_neighbourhood(self):

        curr_sol = self.curr_sol
        neighbourhood = []

        for i in range(0, 30):
            neighbour = deepcopy(curr_sol)
            (set_indices, non_indices) = sep_indices(curr_sol.val)
            rand_S = choice(set_indices)
            rand_N = choice(non_indices)

            neighbour.val[rand_S] = 0
            neighbour.val[rand_N] = 1
            neighbour.fitness = self._score(neighbour)

            path = Path('single_swap', [rand_S, rand_N])
            move = Move(curr_sol, neighbour, path)

            neighbourhood.append(move)

        return neighbourhood


def random_solution(length, num_picked):

    arr = np.array([0] * (length - num_picked) + [1] * num_picked)
    np.random.shuffle(arr)
    return list(arr)


def sep_indices(val):

    set_indices = []
    non_indices = []

    for i in range(0, len(val)):
        if val[i] == 1:
            set_indices += [i]
        else:
            non_indices += [i]

    return (set_indices, non_indices)


if __name__ == "__main__":

    ssn_set, val = max_min_diversity.compute_diverse_set('./temp_ssn_identities.npy',
                                            './temp_ssn_headings.json', 40)

    print(ssn_set)
    ini_sol = Solution(val)

    head = initialise_headings('./temp_ssn_headings.json')
    mat = initialise_matrix('./temp_ssn_identities.npy')

    # ini_sol = Solution(random_solution(241, 40))
    print(ini_sol.val)
    test = MEnzDPTabuSearch(ini_sol, 40, 'double', 10, 1000, opt_tuple=(mat, []))

    print('BEST SCOREEEEEE')
    print(test._score(ini_sol))

    best, score = test.run()

    print(best.val)

    for i in range(0, len(best.val)):
        if best.val[i] == '1':
            print(head[i])

    print(score)


    set = []

    for i in range(0, len(best.val)):
        if best.val[i] == 1:
            set += [i]

    set = sorted([head[x] for x in set])

    print("\nMDP SET:")
    for name in ssn_set:
        print(name + ', ', end='')

    print("\n\nTABUSEARCH SET:")
    for name in set:
        print(name + ', ', end='')
