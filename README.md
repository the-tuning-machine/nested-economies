# Interactions

Simulation d'agents économiques hiérarchiques. Chaque noeud possède des ressources, des skills et des préférences. Les runs testent comment les règles d'échange et de transformation produisent concentration, spécialisation, instabilité ou redistribution.

## Fichiers utiles

- `main.py` : modèle et fonction `run_experiment`.
- `run_experiments.py` : lance la batterie complète de runs avec reprise via `runs/progress.json`.
- `analysis.py` : régénère des métriques et rapports d'analyse à partir des runs.
- `viz.py` : dashboard interactif pour explorer un run.
- `paper.tex` : rapport LaTeX courant.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer les runs

```bash
python run_experiments.py
```

Le script reprend automatiquement les runs déjà terminés grâce à `runs/progress.json`.

## Lancer le dashboard

Interface vide :

```bash
python viz.py
```

Précharger un run :

```bash
python viz.py runs/<run_id>.pkl
```

Ouvrir aussi le navigateur :

```bash
python viz.py --open-browser
```

## Compiler le rapport

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

Le PDF produit est `paper.pdf`.
