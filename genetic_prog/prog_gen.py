import json
import math
import random
import time
from collections import deque
from typing import Optional, List

import numpy as np

from core.ga import GeneticAlgorithm
from core.individual import Individual
from core.population import Population
from genetic_prog.basics import TERMINAL_SET, FUNCTION_SET
from genetic_prog.feaso import FEASO_RANGE_X1, FEASO_RANGE_X2, fEaso, plot_f_x1_x2_pgtree
from genetic_prog.treenode import TerminalNode, OperatorNode
from pgtree import PGTree


def convert_to_math_expression(s):
    import re
    s = s.replace("mul", "*")
    s = s.replace("pow", "**")
    s = s.replace("sub", "-")
    s = s.replace("add", "+")
    s = s.replace("truediv", "/")

    # Убираем лишние пробелы
    s = re.sub(r'\s+', ' ', s).strip()

    return s

MAX_DEPTH = 10
TRAINING_DATA = [
            {'x1': np.pi, 'x2': np.pi},
            {'x1': 5, 'x2': 5},
            {'x1': np.pi/2, 'x2': np.pi/2},
            {'x1': 1.5*np.pi, 'x2': 1.5*np.pi},
            {'x1': 3*np.pi, 'x2': 3*np.pi},
            {'x1': 0, 'x2': 0},
            {'x1': 2.2, 'x2': 4.5},
            {'x1': 1.5, 'x2': 4},
            {'x1': -5, 'x2': 5},
                ]
CHECK_DATA = [{'x1': 3.141592653589793, 'x2': 3.141592653589793}, {'x1': 5, 'x2': 5}, {'x1': 1.5707963267948966, 'x2': 1.5707963267948966}, {'x1': 4.71238898038469, 'x2': 4.71238898038469}, {'x1': 9.42477796076938, 'x2': 9.42477796076938}, {'x1': 0, 'x2': 0}, {'x1': 2.2, 'x2': 4.5}, {'x1': 1.5, 'x2': 4}, {'x1': -5, 'x2': 5}, {'x1': 1.4718549793781799, 'x2': -2.700497646939297}, {'x1': -0.7882949305859466, 'x2': 3.782262865221675}, {'x1': 1.2778469590689756, 'x2': -1.4947758387276466}, {'x1': 2.529712995831516, 'x2': 3.1832039808146675}, {'x1': 3.216063556336371, 'x2': -3.5068081673219376}, {'x1': -4.218341772445876, 'x2': 4.346562018262061}, {'x1': -1.2468677192429576, 'x2': -4.423672148790451}, {'x1': 3.5017612877744124, 'x2': 3.9632741941969325}, {'x1': 1.329202274711431, 'x2': -3.187469316001109}, {'x1': -0.1019168505746082, 'x2': -2.0639853519682894}, {'x1': -0.11142477377290305, 'x2': -1.8969475571232222}, {'x1': -2.628859684452991, 'x2': -2.0264990910096845}, {'x1': 3.52965298113771, 'x2': -0.24160559950122806}, {'x1': 4.1122566921323855, 'x2': 4.3493343800663435}, {'x1': 3.2450133036628905, 'x2': 3.435530042497497}, {'x1': -3.5183810413384387, 'x2': 3.9334311054065303}, {'x1': -0.8642181150560759, 'x2': -2.9885317622320873}, {'x1': -1.4719075102767398, 'x2': -1.5175383614384264}, {'x1': -0.798161125639866, 'x2': 3.7239654850801998}, {'x1': 3.7164385611285553, 'x2': 2.5848359421158875}, {'x1': 4.830304246700493, 'x2': -4.260877792174504}, {'x1': 1.7203519663507247, 'x2': -4.595083824299529}, {'x1': -4.011121621061281, 'x2': -1.9904496030291927}, {'x1': -0.5144558195334481, 'x2': -1.8458868417812226}, {'x1': -4.274127225451733, 'x2': -0.03875062320352107}, {'x1': -2.8910628726259127, 'x2': -4.287336820639105}, {'x1': 3.839186032788847, 'x2': 2.7378167563246016}, {'x1': 1.119394847365709, 'x2': -0.03476089318749942}, {'x1': -2.1566199074758097, 'x2': -3.943027898107976}, {'x1': 3.8684272269403603, 'x2': 0.29310198233697626}, {'x1': 4.133862007762863, 'x2': 2.8584227673522635}, {'x1': 1.8641074344090942, 'x2': -4.630970311804462}, {'x1': -4.479003343658085, 'x2': 2.268740993673851}, {'x1': -4.495315307403067, 'x2': 3.5120690750050247}, {'x1': 2.57676860983858, 'x2': 1.7166890077653267}, {'x1': -0.9310748199844445, 'x2': 0.6176618488381314}, {'x1': 2.945964243573515, 'x2': -3.904794579138191}, {'x1': -4.1296186004172855, 'x2': -1.0623785128143668}, {'x1': 0.46826056395744864, 'x2': 3.61591915500966}, {'x1': 4.565383781537056, 'x2': 1.851050468573244}, {'x1': 0.634556926978183, 'x2': 0.9157529916615514}, {'x1': 4.1656850273346375, 'x2': 0.4989092650248974}, {'x1': 2.302666227321847, 'x2': -0.6178983829229123}, {'x1': 2.3522754464191378, 'x2': -2.1288986815297726}, {'x1': 0.10094473378480195, 'x2': -3.3503092556698943}, {'x1': -4.423194622572394, 'x2': 4.022681276835945}, {'x1': 2.0373189618687615, 'x2': -0.8855774432234229}, {'x1': -0.7347408906784594, 'x2': -0.8067938260370262}, {'x1': 1.3919495744498978, 'x2': -1.108356177849228}, {'x1': 3.831098842395562, 'x2': -4.9569613438486995}, {'x1': -0.08961325723770486, 'x2': -4.096095609901563}, {'x1': -0.939466439802934, 'x2': 4.391970180166915}, {'x1': 3.603509165145457, 'x2': 3.424602833287814}, {'x1': 3.8332946904746485, 'x2': -2.2426855364762135}, {'x1': 0.8210360564728125, 'x2': -0.9560834641655545}, {'x1': 4.1296220736513085, 'x2': -4.519001767812499}, {'x1': -0.2165267301579199, 'x2': 0.6620732906919233}, {'x1': 3.857713818469808, 'x2': -3.818093174062481}, {'x1': -4.400006695591202, 'x2': 2.8855649935904335}, {'x1': -2.973434284295582, 'x2': 4.787372581237799}, {'x1': 2.8348619750389936, 'x2': -4.638849079098231}, {'x1': 4.609247944583261, 'x2': 3.7341844909304562}, {'x1': -3.6185678747751195, 'x2': -2.5313039796511827}, {'x1': -4.291865443539548, 'x2': 2.625246518328245}, {'x1': 3.486758180556988, 'x2': 4.322460593894636}, {'x1': 3.1989587362163725, 'x2': -4.456035392241623}, {'x1': -2.44675600680086, 'x2': -3.464399152306097}, {'x1': 3.354695281343993, 'x2': -2.4650177138515295}, {'x1': -2.5820090065392067, 'x2': 3.516790452019162}, {'x1': 0.9017228123351071, 'x2': -3.1027647882172493}, {'x1': 2.853369913977498, 'x2': -0.719538263083292}, {'x1': -1.8246622731968518, 'x2': 3.185281167773244}, {'x1': -2.892039282691974, 'x2': 4.240690707289666}, {'x1': 4.498952457634864, 'x2': -3.8285870364762875}, {'x1': -1.9790266222053674, 'x2': 1.0689580099597258}, {'x1': -2.0032437386133726, 'x2': -0.09622219910170848}, {'x1': 3.0184567476846063, 'x2': -1.78799077934349}, {'x1': -0.08623857779625688, 'x2': 1.0831263230723192}, {'x1': -4.797889193034591, 'x2': -3.8301087287430757}, {'x1': -4.921428500712036, 'x2': 4.341309465299755}, {'x1': 3.5639741466745143, 'x2': -1.0173382535997755}, {'x1': -4.803352838780659, 'x2': 1.2542423914595542}, {'x1': -1.9407942374371192, 'x2': -0.2433269243354541}, {'x1': 4.138132541277759, 'x2': -4.178604567249142}, {'x1': -0.8940422745788137, 'x2': 3.1796626159521644}, {'x1': -0.9238865666006753, 'x2': 3.311253560116098}, {'x1': -1.7567382330857417, 'x2': 1.3884827396092465}, {'x1': 2.5194749956572204, 'x2': -2.198350090240325}, {'x1': 3.869007629072538, 'x2': 1.6525507972073168}, {'x1': -0.2336795440738184, 'x2': -2.3572515832429963}, {'x1': 3.2793373588989034, 'x2': 3.1314777477853877}]


class PGIndividual:
    def __init__(self, genome: PGTree):
        self.genome = genome

    def mutate(self) -> None:
        """Apply mutation to the individual's genome."""
        # Select a random node in the genome
        random_node = self.genome.get_random_node()

        # If the random node is a TerminalNode, change its value
        if isinstance(random_node, TerminalNode):
            # Choose a new terminal value from the terminal set, ensuring it's different
            new_value = random.choice(TERMINAL_SET)
            random_node.value = new_value

        # If the random node is an OperatorNode, change its operator
        elif isinstance(random_node, OperatorNode):
            # Choose a new operator from the function set, ensuring it's different
            new_operator = random.choice(list(FUNCTION_SET.values()))
            random_node.operator = new_operator

    def crossover(self, other: "PGIndividual") -> "PGIndividual":
        """Perform subtree crossover with another individual."""
        # Create copies of both genomes to avoid modifying the original individuals
        new_genome1 = self.genome.copy()
        new_genome2 = other.genome.copy()

        # Randomly select crossover points in both genomes
        crossover_point1 = new_genome1.get_random_node()
        crossover_point2 = new_genome2.get_random_node()

        # Limit the number of attempts to find compatible nodes
        max_attempts = 10
        attempts = 0

        # Attempt to find compatible crossover points
        while not isinstance(crossover_point1, type(crossover_point2)) and attempts < max_attempts:
            crossover_point1 = new_genome1.get_random_node()
            crossover_point2 = new_genome2.get_random_node()
            attempts += 1

        # If compatible points were found, perform the subtree swap
        if isinstance(crossover_point1, type(crossover_point2)):
            new_genome1.swap_subtrees(crossover_point1, crossover_point2)
        else:
            print("Crossover failed: compatible points not found.")

        # Return the new individual as the offspring
        return PGIndividual(new_genome1)


class PGPopulationFeaso(Population):
    def __init__(self, population_size: int):
        super().__init__(population_size)
        self.etalon = fEaso()
        self.individuals: List[PGIndividual] = []

        self.init_population()


    def init_population(self) -> None:
        for i in range(self.population_size):
            self.individuals.append(PGIndividual(PGTree()))

    def sort_by_fitness(self, reverse:bool=False):
        self.individuals.sort(key=lambda x: self.fitness_function(x), reverse=reverse)

    def fitness_function(self, individual: PGIndividual) -> float:
        """Compare with feaoso function"""

        vars = TRAINING_DATA
        total_td = 0
        total_rd = 0

        for var in vars:
            etalon_val = self.etalon.evaluate(var)
            ind_val = individual.genome.evaluate(var)
            total_td += abs(etalon_val - ind_val)

        for i in range(10):
            v = {'x1': np.random.uniform(*FEASO_RANGE_X1), 'x2': np.random.uniform(*FEASO_RANGE_X2)}
            etalon_val = self.etalon.evaluate(v)
            ind_val = individual.genome.evaluate(v)
            total_rd += abs(etalon_val -ind_val)

        return total_td + abs(individual.genome.evaluate({'x1':math.pi, 'x2':math.pi}) + 1)*10 + total_rd

class PGGeneticAlgorithmFeaso(GeneticAlgorithm):
    def __init__(self, population_size: int, crossover_p: float, mutation_p: float, elite_count: int, debug_info: bool = False):
        super().__init__(population_size, crossover_p, mutation_p)

        self.elite_count = elite_count
        self.population = PGPopulationFeaso(population_size)
        self.elite_individuals: List[PGIndividual] = []
        self.debug_mode = debug_info
        self.steps_info = []
        self.start_time = 0
        self.step = 0
        self.total_info = {'best': None, 'steps': [], 'pc': self.crossover_p, 'pm': self.mutation_p,
                           'n': self.population_size}

    def save_info(self) -> None:
        info = {}
        info['time'] = time.time() - self.start_time
        info['step'] = self.step
        info['best_val'] = self.population.fitness_function(self.get_best())
        self.total_info['steps'].append(info)
        self.total_info['best'] = convert_to_math_expression(str(self.get_best().genome.root))

    def debug(self, info: str):
        if self.debug_mode is False:
            return

        print(f'[DEBUG]: {info}')

    def save_elite(self) -> None:
        self.population.sort_by_fitness()
        self.elite_individuals = self.population[:self.elite_count]

    def clear_invalid(self):
        for ind in self.population.individuals:
            if not ind.genome.is_correct():
                self.population.individuals.remove(ind)
            else:
                ind.genome.prune_tree(MAX_DEPTH)

    def round(self) -> None:
        # 0. Remove incorrects.
        self.clear_invalid()

        # 1. Selection:
        self.debug('Start selection')
        self.selection(10, 5)

        self.debug('Start reproduction')
        # 2. Reproduction.
        self.reproduction()

        # 3. Mutation.
        self.mutation() # mutation inside reproduction

        self.debug('Start reduction')
        # 4. Reduction.
        self.reduction()

        self.step += 1
        self.save_info()


    def selection(self, group_size: int = 2, top_count: int = 1) -> None:
        # Tournament selection;
        tmp_individuals = self.population.individuals.copy()
        random.shuffle(tmp_individuals)

        # Get groups for tournament selection.
        groups = []
        tmp_individuals = deque(tmp_individuals)
        while tmp_individuals:
            group = []
            for i in range(group_size):
                if not tmp_individuals:
                    break
                individ = tmp_individuals.pop()
                group.append(individ)

            groups.append(group)

        tmp_individuals = []
        for g in groups:
            g.sort(key=lambda x: self.population.fitness_function(x))
            tmp_individuals += g[:top_count]

        self.population.individuals = tmp_individuals


    def reproduction(self) -> None:
        p = self.population
        childs = []

        while len(childs) < self.population_size:
            if random.random() <= self.crossover_p:
                parent1, parent2 = random.choices(self.population.individuals, k=2)

                c1 = parent1.crossover(parent2)

                # Mutation for child
                if random.random() <= self.mutation_p:
                    c1.mutate()

                childs.append(c1)
        p.individuals.extend(childs)

    def mutation(self) -> None:
        pass

    def reduction(self) -> None:
        self.population.individuals.extend(self.elite_individuals)
        self.population.sort_by_fitness()
        self.population.individuals = self.population.individuals[:self.population_size]

    def run(self) -> None:
        raise NotImplemented("Still not implemented")

    def run_for(self, iteration: int, filename='result.json'):
        self.start_time = time.time()
        for i in range(iteration):
            self.round()
            print(f'Round {i} finished.')

        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(self.total_info, json_file, ensure_ascii=False, indent=4)

    def get_best(self) -> PGIndividual:
        self.population.sort_by_fitness()
        return self.population.individuals[0]

def expirementA():
    import string
    def generate_random_hash(length=5):
        characters = string.ascii_letters + string.digits
        random_hash = ''.join(random.choice(characters) for _ in range(length))
        return random_hash

    for _ in range(30):
        file_name = f"ga_result_{generate_random_hash()}.json"
        ga = PGGeneticAlgorithmFeaso(400, 0.6, 0.05, 40)
        ga.run_for(100, file_name)

def compare_with_check_data(ga: PGGeneticAlgorithmFeaso):
    best = ga.get_best()
    print(f'Mae: {ga.population.fitness_function(best)}')
    print(f'f: {best.genome.root}')
    print(f'{best.genome.evaluate({"x1": math.pi, "x2": math.pi}) - -1}')
    plot_f_x1_x2_pgtree(best.genome)

    print('Checking data')

    total_error = 0
    i = 0
    for v in CHECK_DATA:
        feaso_val = fEaso().evaluate(v)
        best_val = best.genome.evaluate(v)
        abs_delta = abs(feaso_val - best_val)
        total_error += abs_delta
        print(f'Round {i}: {abs_delta}')
        i += 1

    print(f'MAE:{total_error / len(CHECK_DATA)}')

def experimentB():
    import string
    def generate_random_hash(length=5):
        characters = string.ascii_letters + string.digits
        random_hash = ''.join(random.choice(characters) for _ in range(length))
        return random_hash
    for _ in range(20):
        file_name = f"ga_result_expB_{generate_random_hash()}.json"
        ga = PGGeneticAlgorithmFeaso(2500, 0.6, 0.05, 90)
        ga.run_for(20 , file_name)

        compare_with_check_data(ga)
        ga.get_best().genome.to_dot(file_name+'.dot')

if __name__ == '__main__':
    #expirementA()
    experimentB()


