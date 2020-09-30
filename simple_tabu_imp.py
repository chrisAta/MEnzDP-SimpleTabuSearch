from __future__ import print_function
from .tabstuSearch.TabuSearch import TabuSearch as TabuSearch
from .tabstuSearch.Solution import Solution as Solution
from .tabstuSearch.Path import Path as Path
from .tabstuSearch.Move import Move as Move
from .data_prep import initialise_headings, initialise_matrix

from copy import deepcopy
from random import choice, seed
import numpy as np
import networkx as nx



class MEnzDPTabuSearch(TabuSearch):

    def _score(self, sol, use_clan=False, use_sim=False):

        score = 0
        mat = self.opt_tuple[0]
        clan_threshold = 0.60
        clan_penalty = 0

        sim_list = []

        (set_indices, non_indices) = sep_indices(sol.val)

        for i in range(0, len(set_indices)):
            for j in range(i + 1, len(set_indices)):
                sim = ((1 - mat[set_indices[i], set_indices[j]])) - (mat[set_indices[i], set_indices[j]]/(1 - mat[set_indices[i], set_indices[j]]))

                score += sim

                sim_list.append(mat[set_indices[i], set_indices[j]])

                if mat[set_indices[i], set_indices[j]] >= clan_threshold:
                    clan_penalty += 1

        # if use_clan:
        #     clan_penalty = 1 - (clan_penalty / len(set_indices))
        #     score *= clan_penalty

        if use_sim:
            return score, sim_list

        else:
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

        cls_num = 10

        s_cls = []
        s_cls_tabu = []
        s_cls_nontabu = []
        n_cls = []
        n_cls_tabu = []
        n_cls_nontabu = []

        temp_vals = []
        temp_indices = []

        for i in set_indices:
            temp_vals += [delta[i]]
            temp_indices += [i]

        s_cls = sorted(range(len(temp_vals)), key=lambda i: temp_vals[i])[-cls_num:]
        s_cls = [temp_indices[x] for x in s_cls]

        temp_vals = []
        temp_indices = []

        for i in non_indices:
            temp_vals += [delta[i]]
            temp_indices += [i]

        n_cls = sorted(range(len(temp_vals)), key=lambda i: temp_vals[i])[-cls_num:]
        n_cls = [temp_indices[x] for x in n_cls]

        alpha = []

        for i in s_cls:
            for j in n_cls:
                alpha += [([i, j], delta[i] + delta[j] - ((1 - mat[i, j]))*1.0 - (mat[i, j]/(1 - mat[i, j])))]

        temp_val = -10000000000
        temp_choice = 0

        for i in alpha:
            if i[1] > temp_val:
                temp_val = i[1]
                temp_choice = i

        # print(s_cls, n_cls)
        # print(temp_choice)
        neighbour = deepcopy(curr_sol)

        neighbour.val[temp_choice[0][0]] = 0
        neighbour.val[temp_choice[0][1]] = 1
        neighbour.fitness = self._score(neighbour, True)
        path = Path('single_swap', [temp_choice[0][0], temp_choice[0][1]])
        move = Move(curr_sol, neighbour, path)
        neighbourhood.append(move)

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

        # alpha = []
        #
        # for i in s_cls_tabu:
        #     for j in n_cls_tabu:
        #         alpha += [([i, j], delta[i] + delta[j] - ((mat[i, j])/(1 - mat[i, j])))]
        #
        # if alpha != []:
        #     temp_val = -10000000000
        #     temp_choice = 0
        #
        #     for i in alpha:
        #         if i[1] > temp_val:
        #             temp_choice = i
        #
        #     print(temp_choice)
        #     # print(alpha)
        #     neighbour = deepcopy(curr_sol)
        #     neighbour.val[temp_choice[0][0]] = 0
        #     neighbour.val[temp_choice[0][1]] = 1
        #     neighbour.fitness = self._score(neighbour, True)
        #     path = Path('single_swap', [temp_choice[0][0], temp_choice[0][1]])
        #     move = Move(curr_sol, neighbour, path)
        #
        #     neighbourhood.append(move)
        #
        alpha = []

        for i in s_cls_nontabu:
            for j in n_cls_nontabu:
                alpha += [([i, j], delta[i] + delta[j] - ((1 - mat[i, j]))*1.0 - (mat[i, j]/(1 - mat[i, j])))]

        temp_val = -1000000
        temp_choice = 0

        for i in alpha:
            if i[1] > temp_val:
                temp_val = i[1]
                temp_choice = i

        # print(temp_choice[1])
        # print(temp_choice)
        neighbour = deepcopy(curr_sol)
        neighbour.val[temp_choice[0][0]] = 0
        neighbour.val[temp_choice[0][1]] = 1
        neighbour.fitness = self._score(neighbour, True)
        path = Path('single_swap', [temp_choice[0][0], temp_choice[0][1]])
        move = Move(curr_sol, neighbour, path)

        neighbourhood.append(move)

        # print(neighbourhood[0].path.change)
        # for i in neighbourhood:
        #     print(i)
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

# (mat[indices[0], indices[1]]/(1 - mat[indices[0], indices[1]]))

        delta[indices[0]] = - (delta[indices[0]]) + ((1 - mat[indices[0], indices[1]]))*1.0 + (mat[indices[0], indices[1]]/(1 - mat[indices[0], indices[1]]))   #+penalty
        delta[indices[1]] = - (delta[indices[1]]) + ((1 - mat[indices[0], indices[1]]))*1.0 - (mat[indices[0], indices[1]]/(1 - mat[indices[0], indices[1]])) #-penalty


        # ((mat[indices[0], i])/(1 - mat[indices[0], i]))
        # ((mat[i, indices[1]])/(1 - mat[i, indices[1]]))


        for i in range(0, len(delta)):
            if i in indices:
                continue
            if i in set_indices:
                delta[i] = delta[i] + ((1 - mat[indices[0], i]))*1.0 - ((1 - mat[i, indices[1]]))*1.0 - ((mat[indices[0], i])/(1 - mat[indices[0], i])) + ((mat[i, indices[1]])/(1 - mat[i, indices[1]]))
                continue
            if i in non_indices:

                delta[i] = delta[i] - ((1 - mat[indices[0], i]))*1.0 + ((1 - mat[i, indices[1]]))*1.0 + ((mat[indices[0], i])/(1 - mat[indices[0], i])) - ((mat[i, indices[1]])/(1 - mat[i, indices[1]]))
                continue

        self.opt_tuple[1] = delta


def random_solution(length, num_picked):

    np.random.seed(6102019)

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
            if mat[i, j] == 1:
                continue
            delta[i] += (- (1 - mat[i, j]))*0.5 + (mat[i, j]/(1 - mat[i, j]))


    for i in non_indices:
        for j in set_indices:
            if i == j:
                continue
            if mat[i, j] == 1:
                continue
            delta[i] += ((1 - mat[i, j]))*0.5 - (mat[i, j]/(1 - mat[i, j]))

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


def post_processing(sol, head, mat, delta):

# finds all the cliques at threshold
# keeps one sequence from every clique, deletes the rest
# e.g. if K = 50, and 5 sequences get removed, to keep the original K size, 5 random of the remaining non-picked sequences are swapped in


# alpha += [([i, j], delta[i] + delta[j] - (1 - mat[i, j]))]


    threshold = 0.6

    index_set = set()
    index_list = []

    included_indices = list(np.where(np.array(sol.val) == 1)[0])

    above_thr = np.where(mat >= threshold)
    start_indices = list(above_thr[0])
    end_indices = list(above_thr[1])

    G = nx.Graph()

    for i in range(0, len(start_indices)):

        if start_indices[i] in included_indices and end_indices[i] in included_indices:

            if str(start_indices[i]) + '-' + str(end_indices[i]) not in index_list \
                    and str(end_indices[i])  + '-' + str(start_indices[i]) not in index_list:

                        index_list += [str(start_indices[i]) + '-' + str(end_indices[i])]
                        index_set.add(start_indices[i])
                        index_set.add(end_indices[i])

                        G.add_edge(start_indices[i], end_indices[i])

    start_indices = [int(x.split('-')[0]) for x in index_list]
    end_indices = [int(x.split('-')[1]) for x in index_list]

    cliques = list(nx.find_cliques(G))

    del_indices = []

    for clique in cliques:

        # if len(clique) <= 2:
        #

        count = 0

        while count != len(clique) and len(clique) != 0:

            kept = clique.pop(clique.index(choice(clique)))
            count += 1

            if kept not in del_indices and kept in index_set:

                index_set.remove(kept)
                del_indices += [kept]
                break

    print(index_set)

    # head = list(np.delete(np.array(head), list(index_set)))
    [head.pop(x) for x in list(index_set)]

    new_head = {}

    head_count = 0

    for key in sorted(head.keys()):
        new_head[head_count] = head[key]
        head_count += 1

    zero_indices = list(np.where(np.array(sol.val) == 0)[0])
    # swap_indices = [choice(range(0, len(zero_indices))) for x in index_set]

    changed_indices = []

    for i in index_set:
        max_ind = 0
        max = -50000000

        for j in zero_indices:
            alpha = delta[i] + delta[j] - (1 - mat[i, j])
            if alpha > max and j not in changed_indices:
                max = alpha
                max_ind = j

        sol.val[max_ind] = 1
        sol.val[i] = 0
        changed_indices += [max_ind]

        (set_indices, non_indices) = sep_indices(sol.val)

        delta[i] = - delta[i] + (1 - mat[i, j])
        delta[j] = - delta[j] + (1 - mat[i, j])

        for k in range(0, len(delta)):
            if k in (i, j):
                continue
            if k in set_indices:
                delta[k] = delta[k] + (1 - mat[i, k]) - (1 - mat[k, j])
                continue
            if k in non_indices:
                delta[k] = delta[k] - (1 - mat[i, k]) + (1 - mat[k, j])
                continue

    # swap_indices = sample(range(0, len(zero_indices)), len(del_indices))

    # for i in swap_indices:
    #
    #     sol.val[zero_indices[i]] = 1

    print(len(sol.val))
    sol.val = list(np.delete(np.array(sol.val), list(index_set)))
    print(len(sol.val))

    mat = np.delete(mat, list(index_set), 0)
    mat = np.delete(mat, list(index_set), 1)

    return sol, new_head, mat


def compute_MDP_tabu(mat, head, k, n, preset=None):

    head = initialise_headings(head)
    mat = initialise_matrix(mat)

    seq_len = len(head)

    ini_sol = Solution(random_solution(seq_len, k))
    # ini_sol = Solution(preset)
    print("Initialising Delta")
    delta = initialise_delta(mat, ini_sol)
    results_list = []
    score_list = []
    test = MEnzDPTabuSearch(ini_sol, 50, 'double', 50, 10000, max_wait=50, opt_tuple=[mat, delta])

    best, score, best_progress = test.run()
    new_best_score, sim_list = test._score(best, True, True)

    score_list += [new_best_score]

    for i in range(0, len(best.val)):
        if best.val[i] == '1':
            print(head[i])

    print(score)
    initial_picked_set = []

    for i in range(0, len(best.val)):
        if best.val[i] == 1:
            initial_picked_set += [i]

    initial_picked_set = sorted([head[x] for x in initial_picked_set])
    results_list += [initial_picked_set]

    extra_iterations = n

    new_head = head

    for i in range(0, extra_iterations):

        best, new_head, mat = post_processing(best, new_head, mat, test.opt_tuple[1])
        delta = initialise_delta(mat, best)
        test = MEnzDPTabuSearch(best, 50, 'double', 50, 1000, max_wait=40, opt_tuple=[mat, delta])
        best, score, temp = test.run()
        new_best_score = test._score(best, True)
        new_picked_set = []

        for i in range(0, len(best.val)):
            if best.val[i] == 1:
                new_picked_set += [i]

        new_picked_set = sorted([new_head[x] for x in new_picked_set])
        score_list += [new_best_score]
        results_list += [new_picked_set]

    print("SCORES:")
    print(score_list)

    for i in range(0, len(results_list)):

        print("ITERATION " + str(i) + ' RESULTS')
        print(results_list[i])

    return results_list, score_list, best_progress, sim_list

if __name__ == "__main__":

    compute_MDP_tabu('../Datasets/SSNMatrices/trans1074_identities.npy', '../Datasets/SSNMatrices/trans1074_headings.json', 100)
