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
        mat = self.opt_tuple[0]
        delta = self.opt_tuple[1]

        (set_indices, non_indices) = sep_indices(curr_sol.val)

        set_indices_tabu = []
        set_indices_nontabu = []
        non_indices_tabu = []
        non_indices_nontabu = []

        for i in set_indices:
            if i in self.tabu_list.element_list:
                set_indices_tabu += [i]
            else:
                set_indices_nontabu += [i]

        for i in non_indices:
            if i in self.tabu_list.element_list:
                non_indices_tabu += [i]
            else:
                non_indices_nontabu += [i]

        cls_num = 2

        s_cls_tabu = []
        s_cls_nontabu = []
        n_cls_tabu = []
        n_cls_nontabu = []

        temp_vals = []
        temp_indices = []

        for i in set_indices_tabu:
            temp_vals += [delta[i]]
            temp_indices += [i]

        s_cls_tabu = sorted(range(len(temp_vals)), key=lambda i: temp_vals[i])[-cls_num:]
        s_cls_tabu = [temp_indices[x] for x in s_cls_tabu]

        temp_vals = []
        temp_indices = []

        for i in set_indices_nontabu:
            temp_vals += [delta[i]]
            temp_indices += [i]

        s_cls_nontabu = sorted(range(len(temp_vals)), key=lambda i: temp_vals[i])[-cls_num:]
        s_cls_nontabu = [temp_indices[x] for x in s_cls_nontabu]

        temp_vals = []
        temp_indices = []

        for i in non_indices_tabu:
            temp_vals += [delta[i]]
            temp_indices += [i]

        n_cls_tabu += sorted(range(len(temp_vals)), key=lambda i: temp_vals[i])[-cls_num:]
        n_cls_tabu = [temp_indices[x] for x in n_cls_tabu]

        temp_vals = []
        temp_indices = []

        for i in non_indices_nontabu:
            temp_vals += [delta[i]]
            temp_indices += [i]

        n_cls_nontabu += sorted(range(len(temp_vals)), key=lambda i: temp_vals[i])[-cls_num:]
        n_cls_nontabu = [temp_indices[x] for x in n_cls_nontabu]

        alpha = []

        for i in s_cls_tabu:
            for j in n_cls_tabu:
                alpha += [([i, j], delta[i] + delta[j] - (1 - mat[i, j]))]

        if alpha != []:
            temp_val = -500
            temp_choice = 0

            for i in alpha:
                if i[1] > temp_val:
                    temp_choice = i

            neighbour = deepcopy(curr_sol)
            neighbour.val[temp_choice[0][0]] = 0
            neighbour.val[temp_choice[0][1]] = 1
            neighbour.fitness = self._score(neighbour)
            path = Path('single_swap', [temp_choice[0][0], temp_choice[0][1]])
            move = Move(curr_sol, neighbour, path)

            neighbourhood.append(move)

        alpha = []

        for i in s_cls_nontabu:
            for j in n_cls_nontabu:
                alpha += [([i, j], delta[i] + delta[j] - (1 - mat[i, j]))]

        temp_val = -500
        temp_choice = 0

        for i in alpha:
            if i[1] > temp_val:
                temp_choice = i

        neighbour = deepcopy(curr_sol)

        neighbour.val[temp_choice[0][0]] = 0
        neighbour.val[temp_choice[0][1]] = 1
        neighbour.fitness = self._score(neighbour)
        path = Path('single_swap', [temp_choice[0][0], temp_choice[0][1]])
        move = Move(curr_sol, neighbour, path)

        neighbourhood.append(move)

        return neighbourhood


    # def _create_neighbourhood(self):
    #
    #     curr_sol = self.curr_sol
    #     neighbourhood = []
    #
    #     for i in range(0, 30):
    #         neighbour = deepcopy(curr_sol)
    #         (set_indices, non_indices) = sep_indices(curr_sol.val)
    #         rand_S = choice(set_indices)
    #         rand_N = choice(non_indices)
    #
    #         neighbour.val[rand_S] = 0
    #         neighbour.val[rand_N] = 1
    #         neighbour.fitness = self._score(neighbour)
    #
    #         path = Path('single_swap', [rand_S, rand_N])
    #         move = Move(curr_sol, neighbour, path)
    #
    #         neighbourhood.append(move)
    #
    #     return neighbourhood



    def _post_swap_change(self, move):

        # still thinking about this method in case some post move
        # changes need to be made to something like eq 5 of the memetic algo
        # good place to use the optional tuple
        # for example, if it takes a move as parameter, you can do:
        #
        #
        # Obviously would do this in a loop or something but just to show
        # an example
        #  if i=0 and j=3 were swapped in the algo, then to emulate eq5, i in U and j in Z
        #
        # numpymatrix = self._opt_tuple[0]
        # delta = self._opt_tuple[1]
        # swap_indices = move.path.change
        # delta[0] = - delta[0] + numpymatrix[0,3]
        # delta[3] = - delta[3] + numpymatrix[0,3]
        # delta[1] = delta[1] + numpymatrix[0,1] - numpymatrix[1,3] if [1] in U
        # delta[2] = delta[2] - numpymatrix[0,2] = numpymatrix[1,2] of [2] in Z
        # etc
        #

        mat = self.opt_tuple[0]
        delta = self.opt_tuple[1]
        indices = move.path.change
        temp_sol = move.new_sol

        (set_indices, non_indices) = sep_indices(temp_sol.val)

        delta[indices[0]] = - delta[indices[0]] + (1 - mat[indices[0], indices[1]])
        delta[indices[1]] = - delta[indices[1]] + (1 - mat[indices[0], indices[1]])

        for i in range(0, len(delta)):
            if i in indices:
                continue
            if i in set_indices:
                delta[i] = delta[i] + (1 - mat[indices[0], i]) - (1 - mat[i, indices[1]])
                continue
            if i in non_indices:
                delta[i] = delta[i] - (1 - mat[indices[0], i]) + (1 - mat[i, indices[1]])
                continue

        self.opt_tuple[1] = delta


def random_solution(length, num_picked):

    arr = np.array([0] * (length - num_picked) + [1] * num_picked)
    np.random.shuffle(arr)
    return list(arr)


def initialise_delta(mat, sol):

    delta = [0] * len(sol.val)
    (set_indices, non_indices) = sep_indices(sol.val)

    for i in set_indices:
        for j in set_indices:
            if i == j:
                continue
            delta[i] += -(1 - mat[i, j])

    for i in non_indices:
        for j in set_indices:
            delta[i] += (1 - mat[i, j])

    return delta


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

    print("Running Greedy Max-Min Diversity Solver")
    ssn_set, val = max_min_diversity.compute_diverse_set('./temp_ssn_identities.npy',
                                            './temp_ssn_headings.json', 50)

    head = initialise_headings('./temp_ssn_headings.json')
    mat = initialise_matrix('./temp_ssn_identities.npy')

    ini_sol = Solution(val)

    # ini_sol = Solution(random_solution(241, 50))
    print("Initialising Delta")
    delta = initialise_delta(mat, ini_sol)

    print(ini_sol.val)
    test = MEnzDPTabuSearch(ini_sol, 7, 'double', 20, 1000, opt_tuple=[mat, delta])

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
