from random import choice, randint, random
from string import ascii_lowercase
from TabuSearchGuide import TabuSearch
from copy import deepcopy


class Algorithm(TabuSearch):
    """
    Tries to get a randomly-generated string to match string "clout"
    """
    def _neighborhood(self):
        member = list(self.current)
        neighborhood = []
        for _ in range(50):
            neighbor = deepcopy(member)
            neighbor[randint(0, 4)] = choice(ascii_lowercase)
            # print choice(ascii_lowercase)
            neighbor = ''.join(neighbor)
            neighborhood.append(neighbor)
        return neighborhood

    def _score(self, state):

        return float(sum(state[i] == "clout"[i] for i in range(5)))





def test_algorithm():
    print 'HI'
    algorithm = Algorithm('abcde', 50, 500, max_score=5)
    (member, score) = algorithm.run()

    print member + ' - ' + str(score)


test_algorithm()
