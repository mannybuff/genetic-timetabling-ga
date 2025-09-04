"""
Genetic Timetabling GA
----------------------
A small, self-contained genetic algorithm that assigns instructors and rooms
to a set of courses across time slots, subject to soft constraints.

Derived from an earlier draft (cleaned and modularized) that included:
- activity class and data seeds (time slots, instructors, rooms, courses)
- gene and genome scoring (instructor preference, capacity, conflicts)
- selection, crossover (split/odd/blend), and mutation operators

Author: Manuel Buffa
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from scipy.special import softmax
import copy
import random

# -------------------------
# Seed data (can be swapped)
# -------------------------

TIME_SLOTS = [10, 11, 12, 13, 14, 15]

INSTRUCTORS = ['Lock', 'Glen', 'Banks', 'Richards', 'Shaw', 'Singer', 'Uther', 'Tyler', 'Numen', 'Zeldin']

# [building, room_num, capacity]
ROOMS = [
    ['Slater', '003', 45], ['Roman', '216', 30],
    ['Loft', '206', 75], ['Roman', '201', 50],
    ['Loft', '310', 108],
    ['Beach', '201', 60], ['Beach', '301', 75],
    ['Logos', '325', 450], ['Frank', '119', 60]
]

# course, preferred_instructors, alternate_instructors, expected_enrollment
COURSES = [
    ['SLA101A', ['Glen', 'Lock', 'Banks', 'Zeldin'], ['Numen', 'Richards'], 50],
    ['SLA101B', ['Glen', 'Lock', 'Banks', 'Zeldin'], ['Numen', 'Richards'], 50],
    ['SLA191A', ['Glen', 'Lock', 'Banks', 'Zeldin'], ['Numen', 'Richards'], 50],
    ['SLA191B', ['Glen', 'Lock', 'Banks', 'Zeldin'], ['Numen', 'Richards'], 50],
    ['SLA201',  ['Glen', 'Banks', 'Zeldin', 'Shaw'], ['Numen', 'Richards', 'Singer'], 50],
    ['SLA291',  ['Lock', 'Banks', 'Zeldin', 'Singer'], ['Numen', 'Richards', 'Shaw', 'Tyler'], 50],
    ['SLA303',  ['Glen', 'Zeldin', 'Banks'], ['Numen', 'Singer', 'Shaw'], 60],
    ['SLA304',  ['Glen', 'Banks', 'Tyler'], ['Numen', 'Singer', 'Shaw', 'Richards', 'Uther', 'Zeldin'], 25],
    ['SLA394',  ['Tyler', 'Singer'], ['Richards', 'Zeldin'], 20],
    ['SLA449',  ['Tyler', 'Singer', 'Shaw'], ['Zeldin', 'Uther'], 60],
    ['SLA451',  ['Tyler', 'Singer', 'Shaw'], ['Zeldin', 'Uther', 'Richards', 'Banks'], 100]
]


@dataclass
class Activity:
    course: str
    pref_instr: List[str]
    alt_instr: List[str]
    expected: int
    time_slot: int = 0
    instructor: str = ""
    building: str = ""
    room_num: str = ""
    capacity: int = 0
    fitness: float = 0.0

    def __str__(self):
        return (f"Course: {self.course} Instructor: {self.instructor} "
                f"Time Slot: {self.time_slot}:00 Building: {self.building} "
                f"Room: {self.room_num} Expect: {self.expected} "
                f"Capacity: {self.capacity} Fit: {self.fitness:.3f}")


def seed_activities() -> List[Activity]:
    return [Activity(c, p, a, e, 0, "", "", "", 0, 0.0) for c, p, a, e in COURSES]


# ---------------------
# Genome / Population IO
# ---------------------

def build_population(max_population: int, rng: np.random.Generator | None = None) -> List[List[Activity]]:
    rng = rng or np.random.default_rng()
    base = seed_activities()
    population: List[List[Activity]] = []
    for _ in range(max_population):
        genome = [copy.deepcopy(g) for g in base]
        for gene in genome:
            # time slot
            gene.time_slot = int(rng.choice(TIME_SLOTS))
            # room
            b, rn, cap = ROOMS[int(rng.integers(0, len(ROOMS)))]
            gene.building, gene.room_num, gene.capacity = b, rn, int(cap)
            # instructor
            gene.instructor = str(rng.choice(INSTRUCTORS))
        population.append(genome)
    return population


# -----------------
# Fitness components
# -----------------

def analyze_gene(gene: Activity) -> float:
    # Instructor preference
    if gene.instructor in gene.pref_instr:
        fitness = 0.5
    elif gene.instructor in gene.alt_instr:
        fitness = 0.2
    else:
        fitness = -0.1

    # Capacity suitability
    ratio = (gene.capacity / max(gene.expected, 1))
    if ratio >= 6:
        fitness -= 0.4   # far too large
    elif ratio >= 3:
        fitness -= 0.2   # too large
    elif ratio >= 1:
        fitness += 0.3   # right-sized
    else:
        fitness -= 0.5   # too small
    gene.fitness = fitness
    return fitness


def build_check(g1: Activity, g2: Activity) -> float:
    # Building adjacency preference (Beach/Roman handled specially)
    if g1.building in {'Beach', 'Roman'} and g2.building in {'Beach', 'Roman'}:
        return 0.5 if g1.building == g2.building else -0.4
    return 0.5


def instr_work_check(faculty_counts: Dict[str, int]) -> float:
    # Penalize overloads (>4) or underloads (<2, except Tyler)
    total = 0.0
    for name, cnt in faculty_counts.items():
        if cnt > 4:
            total -= 0.5
        elif cnt < 2 and name != 'Tyler':
            total -= 0.4
    return total


def score_genome(genome: List[Activity]) -> float:
    SLA101 = {'SLA101A', 'SLA101B'}
    SLA191 = {'SLA191A', 'SLA191B'}
    total = 0.0
    faculty_counts: Dict[str, int] = {}

    for i, g in enumerate(genome):
        total += analyze_gene(g)
        faculty_counts[g.instructor] = faculty_counts.get(g.instructor, 0) + 1

        for j in range(i + 1, len(genome)):
            h = genome[j]
            time_diff = abs(g.time_slot - h.time_slot)

            # room conflict
            if g.building == h.building and g.room_num == h.room_num and time_diff == 0:
                total -= 0.5

            # section separation/adjacency
            if g.course in SLA101 and h.course in SLA101:
                total += 0.5 if time_diff >= 4 else -0.5
            elif g.course in SLA191 and h.course in SLA191:
                total += 0.5 if time_diff >= 4 else -0.5
            elif ((g.course in SLA101 and h.course in SLA191) or
                  (g.course in SLA191 and h.course in SLA101)):
                if time_diff == 1:
                    total += build_check(g, h)

            # instructor time conflicts
            if g.instructor == h.instructor:
                if time_diff == 0:
                    total -= 0.2
                elif time_diff == 1:
                    total += build_check(g, h)

    total += instr_work_check(faculty_counts)
    return total


def population_fitness(population: List[List[Activity]]):
    raw_scores = np.array([score_genome(g) for g in population], dtype=float)
    probs = softmax(raw_scores)  # selection probabilities
    best_idx = int(np.argmax(probs))
    best_score = float(np.max(probs))
    mean_score = float(np.mean(probs))
    return probs, mean_score, best_score, best_idx


# ---------------
# GA operators
# ---------------

def cull(population: List[List[Activity]], probs) -> List[List[Activity]]:
    # Sort by probability descending, keep top 30%
    ranked = sorted(range(len(population)), key=lambda i: probs[i], reverse=True)
    keep_n = max(2, int(0.3 * len(population)))
    return [copy.deepcopy(population[i]) for i in ranked[:keep_n]]


def crossover_split(g1: List[Activity], g2: List[Activity]) -> Tuple[List[Activity], List[Activity]]:
    c1, c2 = copy.deepcopy(g1), copy.deepcopy(g2)
    k = min(5, len(c1))
    for i in range(k):
        c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


def crossover_odd(g1: List[Activity], g2: List[Activity]) -> Tuple[List[Activity], List[Activity]]:
    c1, c2 = copy.deepcopy(g1), copy.deepcopy(g2)
    for i in range(1, len(c1), 2):
        c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


def crossover_blend(g1: List[Activity], g2: List[Activity]) -> Tuple[List[Activity], List[Activity]]:
    c1, c2 = copy.deepcopy(g1), copy.deepcopy(g2)
    for i in range(len(c1)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


def make_children(parents: List[List[Activity]], num_children: int) -> List[List[Activity]]:
    children: List[List[Activity]] = []
    # Weighted selection favoring earlier (higher-ranked) parents
    weights = np.array([1 / (i + 1) for i in range(len(parents))], dtype=float)
    weights /= weights.sum()
    while len(children) < num_children:
        i, j = np.random.choice(np.arange(len(parents)), size=2, replace=False, p=weights)
        p1, p2 = parents[i], parents[j]
        choice = random.choice(["split", "odd", "blend"])
        if choice == "split":
            a, b = crossover_split(p1, p2)
        elif choice == "odd":
            a, b = crossover_odd(p1, p2)
        else:
            a, b = crossover_blend(p1, p2)
        children.extend([a, b])
    return children[:num_children]


def mutate(genome: List[Activity]) -> List[Activity]:
    idx = np.random.randint(0, len(genome))
    g = genome[idx]
    g.instructor = random.choice(INSTRUCTORS)
    g.time_slot = random.choice(TIME_SLOTS)
    # optional: shake room choice too
    if random.random() < 0.5:
        b, rn, cap = random.choice(ROOMS)
        g.building, g.room_num, g.capacity = b, rn, cap
    return genome


def mutate_population(population: List[List[Activity]], mutate_rate: float) -> List[List[Activity]]:
    for k in range(len(population)):
        if random.random() < mutate_rate:
            population[k] = mutate(population[k])
    return population


# ---------------
# GA loop
# ---------------

def evolve(
    population_size: int = 80,
    generations: int = 200,
    mutation_rate: float = 0.15,
    rng_seed: int | None = None
):
    if rng_seed is not None:
        np.random.seed(rng_seed)
        random.seed(rng_seed)

    population = build_population(population_size)
    history = []

    for gen in range(generations):
        probs, mean_p, best_p, best_idx = population_fitness(population)
        history.append((gen, float(mean_p), float(best_p)))
        parents = cull(population, probs)
        children = make_children(parents, num_children=max(population_size - len(parents), 0))
        population = parents + children
        population = mutate_population(population, mutation_rate)

    # final evaluation
    probs, mean_p, best_p, best_idx = population_fitness(population)
    best = population[best_idx]
    return {
        "history": history,
        "best_probability": float(best_p),
        "best_index": int(best_idx),
        "best_genome": best
    }


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="Genetic Algorithm Timetabling (toy)")
    p.add_argument("--population", type=int, default=80, help="population size")
    p.add_argument("--generations", type=int, default=200, help="number of generations")
    p.add_argument("--mutation", type=float, default=0.15, help="mutation rate (0..1)")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    args = p.parse_args()

    result = evolve(population_size=args.population, generations=args.generations,
                    mutation_rate=args.mutation, rng_seed=args.seed)
    print("\n=== Best Genome ===")
    for g in result["best_genome"]:
        print(g)
    print("\nBest selection probability:", result["best_probability"])
