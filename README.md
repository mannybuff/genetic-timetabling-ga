# Genetic Timetabling GA

A compact, reproducible **genetic algorithm** that assigns rooms, time slots, and instructors to a small catalog of courses. It demonstrates a clean GA pipeline (fitness, selection, crossover, mutation) with transparent scoring for instructor preferences, room capacity fit, section spacing, and conflict checks.

## Highlights
- Clear **Activity** data model and seeded catalog
- **Fitness** combines instructor preference, capacity fit, conflicts, and workload
- Multiple **crossover** operators (split / odd / blend) and mutation
- Command-line entry point with sensible defaults

## Quickstart
```bash
git clone https://github.com/mannybuff/genetic-timetabling-ga.git
cd genetic-timetabling-ga

python -m venv .venv && source .venv/bin/activate
# (Windows) .venv\Scripts\activate

pip install -r requirements.txt

# Run a short demo
python -m src.ga_scheduler --population 100 --generations 150 --mutation 0.15 --seed 42
```
Artifacts (best genome and scores) are printed to the console. Save formatted tables/plots under `results/` if you extend the project.

## Repo structure
```
genetic-timetabling-ga/
├── src/
│   └── ga_scheduler.py      # GA implementation + CLI
├── notebooks/               # optional EDA or demo runs
├── data/                    # optional external data (if you add any)
├── results/                 # export plots/tables here
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Notes
- The initial data (instructors, rooms, courses, time slots) is kept intentionally small for readability.
- Fitness scoring is **soft**; adjust weights/penalties as your constraints evolve.

## License
MIT © Manuel Buffa, 2025
