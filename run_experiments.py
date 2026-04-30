"""
Orchestrateur de la batterie de runs expérimentaux.

Stratégie des configs :
  - Configs historiques utiles en zero_sum_no_s, en conservant 5 seeds.
  - Sweep utile hors zero_sum_no_s :
      RESOURCE_MODE × TRANSFORM_MODE × ATOMS_ONLY_TRANSFORM × layer_sizes
    → 2 × 4 × 2 × 2 = 32 nouvelles configs, avec 1 seed.
  - runs/progress.json trace les hashs des runs terminés pour permettre la reprise.
"""
import gc
import hashlib
import json
import os

import torch
from tqdm import tqdm

from main import run_experiment

# ── Constantes partagées ───────────────────────────────────────────────────────
_FIXED = dict(
    n_steps               = 20000,
    lr                    = 1e-3,
    N_REACTIONS           = 10,
    UTILITY_MODE          = 'self',
    MARKET_STRENGTH       = 1.0,
    RESOURCE_MODE         = 'zero_sum_no_s',
    STRICT_COBB_DOUGLAS   = True,
    NO_PARENT_UTILITY     = False,
    NO_PARENT_SKILL       = False,
    NO_RESOURCE_SELECTION = False,
    NO_SKILL_SELECTION    = False,
    NO_UTILITY_SELECTION  = False,
    FREEZE_SKILLS         = False,
    FREEZE_UTILITIES      = False,
    alpha_mem             = 0.9,
    lambda_rec            = 0.1,
)

# Défauts pour les variables expérimentales
_DEFAULTS = dict(
    n_res                = 5,
    MODE                 = 'reciprocity',
    TRANSFORM_MODE       = 'mass_action',
    ATOMS_ONLY_TRANSFORM = True,
    BUDGET_MODE          = 'joint',
)

# Configs historiques utiles à conserver dans le lanceur.
# Les variantes markov/stoichiometric/leontief en zero_sum_no_s sont exclues :
# elles ont été jugées redondantes avec mass_action dans ce mode.
_BASE_VARIANTS = [
    {},                                            # 1. config par défaut
    dict(n_res=15, n_steps=40000),                 # 2. plus de ressources (plus lent → 2× steps)
    dict(MODE='market'),                           # 3. mode market
    dict(ATOMS_ONLY_TRANSFORM=False),              # 4. tous les nœuds transforment
    dict(BUDGET_MODE='split'),                     # 5. budget séparé
]

_RESOURCE_MODES = [
    'zero_sum_stoch_s',
    'free',
]

_TRANSFORM_MODES = [
    'mass_action',
    'markov',
    'stoichiometric',
    'leontief',
]

_ATOMS_ONLY_VALUES = [
    True,
    False,
]

_LAYER_SIZES = [
    [10] * 10,
    [32, 16, 8, 4, 2, 1],
]

# ── Construction de la liste complète de configs ──────────────────────────────
def _build_jobs():
    jobs = []

    # Contrôles historiques utiles en zero_sum_no_s : 5 seeds, déjà présents
    # dans progress.json sauf relance volontaire.
    for layer_sizes in _LAYER_SIZES:
        for variant in _BASE_VARIANTS:
            cfg = {**_FIXED, **_DEFAULTS, **variant, 'layer_sizes': layer_sizes}
            jobs.append((cfg, 5))

    # Sweep où TRANSFORM_MODE et ATOMS_ONLY_TRANSFORM ont un effet réel.
    for layer_sizes in _LAYER_SIZES:
        for resource_mode in _RESOURCE_MODES:
            for transform_mode in _TRANSFORM_MODES:
                for atoms_only in _ATOMS_ONLY_VALUES:
                    cfg = {
                        **_FIXED,
                        **_DEFAULTS,
                        'RESOURCE_MODE': resource_mode,
                        'TRANSFORM_MODE': transform_mode,
                        'ATOMS_ONLY_TRANSFORM': atoms_only,
                        'layer_sizes': layer_sizes,
                    }
                    jobs.append((cfg, 1))
    return jobs


def _build_configs():
    return [cfg for cfg, _n_samples in _build_jobs()]


def _config_hash(cfg: dict) -> str:
    return hashlib.md5(str(sorted(cfg.items())).encode()).hexdigest()[:6]


# ── Gestion de la progression ──────────────────────────────────────────────────
_PROGRESS_PATH = os.path.join('runs', 'progress.json')


def _load_progress() -> dict:
    if os.path.exists(_PROGRESS_PATH):
        with open(_PROGRESS_PATH) as f:
            return json.load(f)
    return {}


def _save_progress(progress: dict):
    os.makedirs('runs', exist_ok=True)
    tmp = _PROGRESS_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, _PROGRESS_PATH)


# ── Point d'entrée ─────────────────────────────────────────────────────────────
def main():
    torch.set_default_dtype(torch.float64)

    jobs     = _build_jobs()
    progress = _load_progress()
    total    = sum(n_samples for _cfg, n_samples in jobs)

    print(f"\n{'='*60}")
    print(f"  Batterie de {len(jobs)} configs = {total} runs planifiés")
    print(f"  Reprise depuis : {_PROGRESS_PATH}")
    print(f"{'='*60}\n")

    with tqdm(total=total, unit='run', dynamic_ncols=True) as pbar:
        for cfg, n_samples in jobs:
            chash = _config_hash(cfg)
            postfix = (
                f"RES={cfg['RESOURCE_MODE']} n_res={cfg['n_res']} MODE={cfg['MODE']} "
                f"TRANSFORM={cfg['TRANSFORM_MODE']} "
                f"ATOMS_ONLY={cfg['ATOMS_ONLY_TRANSFORM']} "
                f"BUDGET={cfg['BUDGET_MODE']}"
            )

            for seed in range(n_samples):
                key = f"{chash}_{seed}"
                pbar.set_description(f"{chash} s{seed}")
                pbar.set_postfix_str(postfix, refresh=False)

                if key in progress:
                    pbar.update(1)
                    continue

                run_id = run_experiment({**cfg, 'seed': seed}, verbose=False)

                progress[key] = run_id
                _save_progress(progress)
                tqdm.write(f"  DONE  {chash} s{seed}  →  {run_id}")

                pbar.update(1)
                gc.collect()

    print(f"\n{'='*60}")
    print(f"  Tous les runs sont terminés.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
