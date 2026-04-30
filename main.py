import gc
import os
import json
import pickle
import hashlib
import datetime
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_finite(name, x):
    """Fail fast dès qu'une quantité non finie apparaît dans la dynamique."""
    if torch.isfinite(x).all():
        return

    bad = ~torch.isfinite(x)
    idx = bad.nonzero(as_tuple=False)[0].tolist()
    value = x[tuple(idx)].item()
    raise RuntimeError(f"Non-finite value detected in {name} at index {idx}: {value}")


def _safe_row_normalize(x, dim, eps=1e-8):
    """Normalisation robuste avec repli uniforme si une ligne devient quasi nulle."""
    _assert_finite("normalize_input", x)
    denom = x.sum(dim=dim, keepdim=True)
    _assert_finite("normalize_denom", denom)
    normalized = x / denom.clamp(min=eps)

    n = x.size(dim)
    uniform = torch.full_like(x, 1.0 / n)
    valid = denom > eps
    return torch.where(valid.expand_as(x), normalized, uniform)


def _sanitize_nonfinite_grads(parameters):
    """
    Remplace les gradients non finis par 0 avant le clipping.
    Retourne le nombre de tenseurs de gradient modifiés.
    """
    n_fixed = 0
    for p in parameters:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            n_fixed += 1
    return n_fixed


def _group_score(logits):
    """
    Score agrégé d'un groupe de destinataires, indépendant de sa cardinalité.
    Évite qu'un groupe avec plus de voisins capte mécaniquement plus de masse.
    """
    if logits.size(1) == 0:
        raise ValueError("group logits must contain at least one recipient")
    return torch.logsumexp(logits, dim=1) - torch.log(
        torch.tensor(logits.size(1), dtype=logits.dtype, device=logits.device)
    )


class EconomicHierarchicalDAG(nn.Module):
    def __init__(self, layer_sizes, n_res,
                 mode='baseline', alpha=0.9, lam=0.1,
                 resource_mode='free', no_parent_utility=False, no_parent_skill=False,
                 no_resource_selection=False, no_skill_selection=False,
                 no_utility_selection=False, resource_update_rate=0.5,
                 freeze_skills=False, freeze_utilities=False,
                 strict_cobb_douglas=True,
                 transform_mode='markov', n_reactions=10,
                 atoms_only_transform=False, utility_mode='self',
                 market_strength=1.0, budget_mode='joint'):
        """
        Initialise le système économique hiérarchique.

        Args:
            layer_sizes       (list of int): Nombre de nœuds par couche [N_0, N_1, ..., N_L-1].
            n_res             (int): Nombre de ressources distinctes.
            mode              (str): 'baseline' | 'reciprocity'
            alpha             (float): Décroissance EMA des mémoires d'arête (reciprocity).
            lam               (float): Poids du bonus de réciprocité (reciprocity).
            resource_mode     (str): 'free'             — S transforme librement les ressources
                                     'zero_sum_no_s'    — S fixé à I, échanges purs (somme nulle)
                                     'zero_sum_stoch_s' — S row-stochastic, masse conservée par nœud
            no_parent_utility (bool): Annule les flux d'utilité extrinsèques.
            no_parent_skill   (bool): Annule les flux de skills extrinsèques.
            no_resource_selection (bool): Conservé pour compatibilité.
            no_skill_selection    (bool): Proportions S moyennées sur n_res.
            no_utility_selection  (bool): Proportions U moyennées sur n_res.
            resource_update_rate (float): Sous-relaxation de la mise à jour des ressources.
            freeze_skills       (bool): Figement des skills S.
            freeze_utilities    (bool): Figement des utilités u.
            strict_cobb_douglas (bool): Si True, utilise log(r). Si False, utilise log1p(r).
            transform_mode      (str): 'markov'         — einsum(r, S) original (chaîne de Markov)
                                       'stoichiometric' — réactions avec réactif limitant (min)
                                       'leontief'       — recette de crafting (min des ratios)
                                       'mass_action'    — cinétique multiplicative log-linéaire
            n_reactions         (int): Nombre de réactions K (transform_mode='stoichiometric').
            atoms_only_transform (bool): Si True, seuls les nœuds l=0 (atomes) appliquent
                                        _transform_resources(); les méta-agents redistribuent
                                        sans transformer (r_transformed = r).
            utility_mode        (str): 'self'      — Cobb-Douglas sur ressources propres.
                                       'aggregate' — ajoute la contribution pondérée des enfants:
                                                     r_eval_im = r_im + Σ_k α_{k→i,m} · r_km,
                                                     α_{k→i,m} = P_up[l-1][k,i,m] / Σ_j P_up[l-1][k,j,m].
            market_strength     (float): Intensité du signal marché (mode='market').
                                         Échelle du bonus d'alignement offre/demande ajouté aux logits R.
            budget_mode         (str): 'joint' — un seul softmax [keep, up, down] : les directions
                                                  se font concurrence dans un budget partagé (défaut).
                                       'split' — deux softmax indépendants [keep, up] et [keep, down] :
                                                  chaque direction a son propre budget, pas de
                                                  concurrence entre monter et descendre.
        """
        super().__init__()
        self.layer_sizes       = layer_sizes
        self.n_layers          = len(layer_sizes)
        self.n_res             = n_res
        self.mode              = mode
        self.alpha             = alpha
        self.lam               = lam
        self.resource_mode     = resource_mode
        self.no_parent_utility = no_parent_utility
        self.no_parent_skill      = no_parent_skill
        self.no_resource_selection = no_resource_selection
        self.no_skill_selection    = no_skill_selection
        self.no_utility_selection  = no_utility_selection
        self.resource_update_rate  = resource_update_rate
        self.freeze_skills         = freeze_skills
        self.freeze_utilities      = freeze_utilities
        self.strict_cobb_douglas   = strict_cobb_douglas
        self.transform_mode        = transform_mode
        self.n_reactions           = n_reactions
        self.atoms_only_transform  = atoms_only_transform
        self.utility_mode          = utility_mode
        self.market_strength       = market_strength
        self.budget_mode           = budget_mode
        
        # --- 1. PARAMÈTRES D'OPTIMISATION (Variables d'interaction sortantes) ---
        self.logits_R_up = nn.ParameterList()
        self.logits_R_down = nn.ParameterList()
        self.logits_S_up = nn.ParameterList()
        self.logits_S_down = nn.ParameterList()
        self.logits_U_up = nn.ParameterList()
        self.logits_U_down = nn.ParameterList()
        
        for l in range(self.n_layers):
            N_l = layer_sizes[l]
            N_up = layer_sizes[l+1] if l < self.n_layers - 1 else 0
            N_down = layer_sizes[l-1] if l > 0 else 0
            
            self.logits_R_up.append(nn.Parameter(torch.randn(N_l, N_up, n_res) * 0.1))
            self.logits_R_down.append(nn.Parameter(torch.randn(N_l, N_down, n_res) * 0.1))

            self.logits_S_up.append(nn.Parameter(torch.randn(N_l, N_up, n_res) * 0.1))
            self.logits_S_down.append(nn.Parameter(torch.randn(N_l, N_down, n_res) * 0.1))

            self.logits_U_up.append(nn.Parameter(torch.randn(N_l, N_up, n_res) * 0.1))
            self.logits_U_down.append(nn.Parameter(torch.randn(N_l, N_down, n_res) * 0.1))
            
        self.logits_keep_R = nn.ParameterList([nn.Parameter(torch.full((N_l, 1, n_res), 5.0)) for N_l in layer_sizes])
        self.logits_keep_S = nn.ParameterList([nn.Parameter(torch.full((N_l, 1, n_res), 5.0)) for N_l in layer_sizes])
        self.logits_keep_U = nn.ParameterList([nn.Parameter(torch.full((N_l, 1, n_res), 5.0)) for N_l in layer_sizes])

        # --- 2. ÉTATS DES NOEUDS (Mémoire du système) ---
        for l, N_l in enumerate(layer_sizes):
            self.register_buffer(f'r_{l}',
                torch.distributions.Exponential(torch.ones(N_l, n_res)).sample())
            self.register_buffer(f'S_{l}',
                F.softmax(torch.randn(N_l, n_res, n_res), dim=2))
            self.register_buffer(f'u_{l}',
                F.softmax(torch.randn(N_l, n_res), dim=1))

        # --- 3. PARAMÈTRES SPÉCIFIQUES AU MODE DE TRANSFORMATION ---
        if transform_mode == 'stoichiometric':
            # Matrices de réaction globales (positives via softplus)
            self.W_in  = nn.Parameter(torch.randn(n_reactions, n_res) * 0.1)
            self.W_out = nn.Parameter(torch.randn(n_reactions, n_res) * 0.1)
            # Vitesses d'activation par nœud : (N_l, K)
            self.v_raw = nn.ParameterList([
                nn.Parameter(torch.zeros(N_l, n_reactions)) for N_l in layer_sizes
            ])
        elif transform_mode == 'leontief':
            # Matrice des prérequis globale : C[j,m] = quantité de j pour produire 1 de m
            self.C_raw = nn.Parameter(torch.randn(n_res, n_res) * 0.1)
            # Efficacité par nœud et par ressource : (N_l, N)
            self.eff_raw = nn.ParameterList([
                nn.Parameter(torch.zeros(N_l, n_res)) for N_l in layer_sizes
            ])
        # 'markov' utilise S existant ; 'mass_action' utilise S en log-espace

        # --- 4. COMPOSANTS SPÉCIFIQUES AU MODE D'OPTIMISATION ---
        if self.mode == 'reciprocity':
            for l in range(self.n_layers):
                N_l   = layer_sizes[l]
                N_up  = layer_sizes[l + 1] if l < self.n_layers - 1 else 0
                N_dn  = layer_sizes[l - 1] if l > 0 else 0
                r_l   = getattr(self, f'r_{l}')
                if N_up > 0:
                    self.register_buffer(f'M_up_{l}',
                        (r_l / N_up).unsqueeze(1).expand(N_l, N_up, n_res).clone())
                if N_dn > 0:
                    self.register_buffer(f'M_down_{l}',
                        (r_l / N_dn).unsqueeze(1).expand(N_l, N_dn, n_res).clone())

    def get_proportions(self, l, type_var, bonus_up=None, bonus_down=None):
        """
        Convertit les logits en proportions d'allocation valides (somme <= 1) via Softmax.

        bonus_up/bonus_down : tenseurs additifs optionnels (même shape que les logits)
            utilisés par le mode 'market' pour biaiser le routage vers des partenaires
            alignés — calculés depuis les états courants (detachés, pas de gradient).
        """
        if type_var == 'R':
            log_up, log_down, log_keep = self.logits_R_up[l], self.logits_R_down[l], self.logits_keep_R[l]
        elif type_var == 'S':
            log_up, log_down, log_keep = self.logits_S_up[l], self.logits_S_down[l], self.logits_keep_S[l]
        else: # 'U'
            log_up, log_down, log_keep = self.logits_U_up[l], self.logits_U_down[l], self.logits_keep_U[l]

        # Appliquer les bonus marché aux logits (détachés → pas de grad supplémentaire)
        if bonus_up is not None:
            log_up = log_up + bonus_up
        if bonus_down is not None:
            log_down = log_down + bonus_down

        N_up = self.layer_sizes[l+1] if l < self.n_layers - 1 else 0
        N_down = self.layer_sizes[l-1] if l > 0 else 0

        log_keep_sq = log_keep.squeeze(1)  # (N_l, n_res)

        if self.budget_mode == 'split' and N_up > 0 and N_down > 0:
            # ── Budget séparé : chaque direction a son propre softmax [keep, dir] ──
            up_b = F.softmax(torch.stack([log_keep_sq, _group_score(log_up)], dim=1), dim=1)
            up_b = _safe_row_normalize(up_b, dim=1)
            dn_b = F.softmax(torch.stack([log_keep_sq, _group_score(log_down)], dim=1), dim=1)
            dn_b = _safe_row_normalize(dn_b, dim=1)
            _assert_finite(f"budget_up_{type_var}_l{l}", up_b)
            _assert_finite(f"budget_dn_{type_var}_l{l}", dn_b)

            up_dist = _safe_row_normalize(F.softmax(log_up, dim=1), dim=1)
            prop_up = up_dist * up_b[:, 1:2, :]   # fraction allouée à la direction haut

            dn_dist = _safe_row_normalize(F.softmax(log_down, dim=1), dim=1)
            prop_down = dn_dist * dn_b[:, 1:2, :]  # fraction allouée à la direction bas

        else:
            # ── Budget joint (défaut) : un seul softmax [keep, up, down] ──────────
            budget_logits = [log_keep_sq]
            groups = ['keep']
            if N_up > 0:
                budget_logits.append(_group_score(log_up))
                groups.append('up')
            if N_down > 0:
                budget_logits.append(_group_score(log_down))
                groups.append('down')

            budget_tensor = torch.stack(budget_logits, dim=1)
            _assert_finite(f"budget_logits_{type_var}_l{l}", budget_tensor)
            budgets = _safe_row_normalize(F.softmax(budget_tensor, dim=1), dim=1)

            group_budget = {
                name: budgets[:, idx, :].unsqueeze(1)
                for idx, name in enumerate(groups)
            }

            prop_up = None
            if N_up > 0:
                prop_up = _safe_row_normalize(F.softmax(log_up, dim=1), dim=1) * group_budget['up']

            prop_down = None
            if N_down > 0:
                prop_down = _safe_row_normalize(F.softmax(log_down, dim=1), dim=1) * group_budget['down']

        if self.no_skill_selection and type_var == 'S':
            if prop_up is not None:
                prop_up = prop_up.mean(dim=2, keepdim=True).expand_as(prop_up)
            if prop_down is not None:
                prop_down = prop_down.mean(dim=2, keepdim=True).expand_as(prop_down)

        if self.no_utility_selection and type_var == 'U':
            if prop_up is not None:
                prop_up = prop_up.mean(dim=2, keepdim=True).expand_as(prop_up)
            if prop_down is not None:
                prop_down = prop_down.mean(dim=2, keepdim=True).expand_as(prop_down)

        return prop_up, prop_down

    def get_state(self, l):
        return getattr(self, f'r_{l}'), getattr(self, f'S_{l}'), getattr(self, f'u_{l}')

    def set_state(self, l, r, S, u):
        setattr(self, f'r_{l}', r.detach()) 
        setattr(self, f'S_{l}', S.detach())
        setattr(self, f'u_{l}', u.detach())

    def _enforce_global_resource_conservation(self, prev_states, new_states):
        """
        Recale la masse totale des ressources dans les modes conservatifs.
        """
        if self.resource_mode != 'zero_sum_stoch_s':
            return new_states

        prev_total = sum(r.sum() for r, _, _ in prev_states)
        new_total = sum(r.sum() for r, _, _ in new_states)
        _assert_finite("prev_total_resources", prev_total)
        _assert_finite("new_total_resources", new_total)
        scale = prev_total / new_total.clamp(min=1e-12)

        conserved = [(r * scale, S, u) for r, S, u in new_states]
        for l, (r, S, u) in enumerate(conserved):
            _assert_finite(f"conserved_r_l{l}", r)
            _assert_finite(f"conserved_S_l{l}", S)
            _assert_finite(f"conserved_u_l{l}", u)
        return conserved

    def _relax_resources(self, r_old, r_candidate):
        """
        Sous-relaxation pour casser les oscillations de période 2.
        """
        beta = self.resource_update_rate
        if beta >= 1.0:
            return r_candidate
        return (1.0 - beta) * r_old + beta * r_candidate

    def _transform_resources(self, l, r, S):
        """
        Transformation interne des ressources d'un nœud.

        Retourne r_transformed à utiliser dans : r_new = F.relu(r_transformed + r_net).

        Modes :
          'markov'         — einsum multiplicatif via S (chaîne de Markov, original).
          'stoichiometric' — réactions chimiques avec réactif limitant (opérateur min).
          'leontief'       — recette de crafting : min des ratios stock/prérequis.
          'mass_action'    — cinétique log-linéaire (Cobb-Douglas multiplicatif).
        """
        eps = 1e-8

        if self.transform_mode == 'markov':
            r_out = torch.einsum('in, inm -> im', r, S)
            # Cas instable observé : free + markov sur toutes les couches.
            if self.resource_mode == 'free' and not self.atoms_only_transform:
                r_out = r_out.clamp(max=torch.exp(torch.tensor(80.0, dtype=r_out.dtype, device=r_out.device)))
            return r_out

        elif self.transform_mode == 'stoichiometric':
            W_in  = F.softplus(self.W_in)   # (K, N), positif
            W_out = F.softplus(self.W_out)  # (K, N), positif
            v     = F.softplus(self.v_raw[l])  # (N_l, K), positif

            # Vitesse maximale par réaction : réactif limitant
            # ratios[i, k, n] = r[i, n] / W_in[k, n]
            ratios = r.unsqueeze(1) / (W_in.unsqueeze(0) + eps)  # (N_l, K, N)
            # Les ressources non consommées (W_in ≈ 0) ne limitent pas
            mask = W_in.unsqueeze(0) < eps
            ratios = ratios.masked_fill(mask, 1e9)
            v_max = ratios.min(dim=2).values  # (N_l, K)

            v_actual = torch.min(v, v_max)
            _assert_finite(f"v_actual_l{l}", v_actual)

            # Variation nette : consommation + production
            delta_r = torch.einsum('ik, km -> im', v_actual, W_out - W_in)
            return r + delta_r

        elif self.transform_mode == 'leontief':
            C   = F.softplus(self.C_raw)          # (N, N), positif
            eff = torch.sigmoid(self.eff_raw[l])  # (N_l, N)

            # scale[i, m] = min_j( r[i,j] / C[j,m] ) où C[j,m] > 0
            # ratios[i, j, m] = r[i, j] / C[j, m]
            ratios = r.unsqueeze(2) / (C.unsqueeze(0) + eps)  # (N_l, N, N)
            mask = C.unsqueeze(0) < eps
            ratios = ratios.masked_fill(mask, 1e9)
            scale = ratios.min(dim=1).values  # (N_l, N)
            # Cas instable observé : free + leontief limité aux atomes peut faire
            # diverger l'échelle Leontief de la couche 0 avant même l'utilité.
            if (
                self.resource_mode == 'free'
                and self.transform_mode == 'leontief'
                and self.atoms_only_transform
            ):
                scale = scale.clamp(max=torch.exp(torch.tensor(30.0, dtype=scale.dtype, device=scale.device)))

            _assert_finite(f"leontief_scale_l{l}", scale)
            return scale * eff

        elif self.transform_mode == 'mass_action':
            # log(r_out[m]) = sum_n S[n,m] * log(r[n])  →  cinétique multiplicative
            log_r = torch.log(r.clamp(min=eps))  # (N_l, N)
            log_r_out = torch.einsum('in, inm -> im', log_r, S)
            # Cas instable observé : free + mass_action sur toutes les couches.
            if self.resource_mode == 'free' and not self.atoms_only_transform:
                log_r_out = log_r_out.clamp(max=80.0)
            return torch.exp(log_r_out)

        else:
            raise ValueError(f"transform_mode inconnu : {self.transform_mode!r}")

    def _compute_market_bonuses(self):
        """
        [Market] Calcule les bonus d'alignement offre/demande pour biaiser le routage R.

        Pour l'envoi vers le haut (i → j au niveau l+1) :
            bonus_up[i,j,m] = u_l[i,m] · r_{l+1}[j,m]
            → i préfère les parents j qui possèdent les ressources que i valorise.

        Pour l'envoi vers le bas (i → j au niveau l-1) :
            bonus_down[i,j,m] = r_l[i,m] · u_{l-1}[j,m]
            → i préfère les enfants j qui ont besoin de ce que i peut offrir.

        Les bonus sont normalisés par nœud (max → 1) puis mis à l'échelle par market_strength,
        de sorte qu'ils restent comparables aux logits quelle que soit la magnitude des ressources.
        """
        bonus_up_list   = []
        bonus_down_list = []

        for l in range(self.n_layers):
            r_l = getattr(self, f'r_{l}').detach()  # (N_l, n_res)
            u_l = getattr(self, f'u_{l}').detach()  # (N_l, n_res)

            if l < self.n_layers - 1:
                r_up = getattr(self, f'r_{l+1}').detach()      # (N_{l+1}, n_res)
                # bonus[i,j,m] = u_l[i,m] * r_up[j,m]
                b_up = torch.einsum('im, jm -> ijm', u_l, r_up)
                # Normalisation par nœud émetteur : max sur (j,m) → [0, market_strength]
                scale = b_up.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                bonus_up_list.append(self.market_strength * b_up / scale)
            else:
                bonus_up_list.append(None)

            if l > 0:
                u_dn = getattr(self, f'u_{l-1}').detach()      # (N_{l-1}, n_res)
                # bonus[i,j,m] = r_l[i,m] * u_dn[j,m]
                b_dn = torch.einsum('im, jm -> ijm', r_l, u_dn)
                scale = b_dn.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                bonus_down_list.append(self.market_strength * b_dn / scale)
            else:
                bonus_down_list.append(None)

        return bonus_up_list, bonus_down_list

    def forward(self):
        """
        Effectue un pas de temps de la dynamique (t -> t+1).
        """
        new_states = []
        
        # 1. Calculer toutes les proportions d'envoi
        if self.mode == 'market':
            bonuses_up, bonuses_down = self._compute_market_bonuses()
            P_R_up, P_R_down = zip(*[
                self.get_proportions(l, 'R',
                                     bonus_up=bonuses_up[l],
                                     bonus_down=bonuses_down[l])
                for l in range(self.n_layers)
            ])
        else:
            P_R_up, P_R_down = zip(*[self.get_proportions(l, 'R') for l in range(self.n_layers)])
        P_S_up, P_S_down = zip(*[self.get_proportions(l, 'S') for l in range(self.n_layers)])
        P_U_up, P_U_down = zip(*[self.get_proportions(l, 'U') for l in range(self.n_layers)])

        # 2. Mettre à jour chaque couche
        for l in range(self.n_layers):
            r, S, u = self.get_state(l)
            N_l = self.layer_sizes[l]

            # --- FLUX INTRINSÈQUES (Depuis/Vers les Enfants en l-1) ---
            if l > 0:
                r_prev, S_prev, u_prev = self.get_state(l-1)
                r_int_in = torch.einsum('kin, kn -> in', P_R_up[l-1].detach(), r_prev)
                S_int_in = torch.einsum('kin, knm -> inm', P_S_up[l-1].detach(), S_prev)
                u_int_in = torch.einsum('kin, kn -> in', P_U_up[l-1].detach(), u_prev)

                r_int_out = torch.einsum('ikn, in -> in', P_R_down[l], r)
                S_int_out = torch.einsum('ikn, inm -> inm', P_S_down[l], S)
                u_int_out = torch.einsum('ikn, in -> in', P_U_down[l], u)
            else:
                r_int_in = r_int_out = torch.zeros_like(r)
                S_int_in = S_int_out = torch.zeros_like(S)
                u_int_in = u_int_out = torch.zeros_like(u)

            # --- FLUX EXTRINSÈQUES (Depuis/Vers les Parents en l+1) ---
            if l < self.n_layers - 1:
                r_next, S_next, u_next = self.get_state(l+1)
                r_ext_in = torch.einsum('jin, jn -> in', P_R_down[l+1].detach(), r_next)
                S_ext_in = (torch.zeros_like(S) if self.no_parent_skill
                            else torch.einsum('jin, jnm -> inm', P_S_down[l+1].detach(), S_next))
                u_ext_in = (torch.zeros_like(u) if self.no_parent_utility
                            else torch.einsum('jin, jn -> in', P_U_down[l+1].detach(), u_next))

                r_ext_out = torch.einsum('ijn, in -> in', P_R_up[l], r)
                S_ext_out = (torch.zeros_like(S) if self.no_parent_skill
                             else torch.einsum('ijn, inm -> inm', P_S_up[l], S))
                u_ext_out = (torch.zeros_like(u) if self.no_parent_utility
                             else torch.einsum('ijn, in -> in', P_U_up[l], u))
            else:
                r_ext_in = r_ext_out = torch.zeros_like(r)
                S_ext_in = S_ext_out = torch.zeros_like(S)
                u_ext_in = u_ext_out = torch.zeros_like(u)

            # --- DYNAMIQUES GLOBALES (Mise à jour de t à t+1) ---
            r_net = (r_int_in - r_int_out) + (r_ext_in - r_ext_out)
            _assert_finite(f"r_net_l{l}", r_net)

            if self.resource_mode == 'zero_sum_no_s':
                r_new = F.relu(r + r_net)
                r_new = self._relax_resources(r, r_new)
                S_new = S
            else:
                if self.atoms_only_transform and l > 0:
                    r_transformed = r
                else:
                    r_transformed = self._transform_resources(l, r, S)
                _assert_finite(f"r_transformed_l{l}", r_transformed)
                r_new = F.relu(r_transformed + r_net)
                r_new = self._relax_resources(r, r_new)
                S_net = (S_int_in - S_int_out) + (S_ext_in - S_ext_out)
                _assert_finite(f"S_net_l{l}", S_net)
                S_new = F.relu(S + S_net)
                if self.resource_mode == 'zero_sum_stoch_s':
                    S_new = _safe_row_normalize(S_new, dim=2)
            if self.freeze_skills:
                S_new = S

            u_net = (u_int_in - u_int_out) + (u_ext_in - u_ext_out)
            _assert_finite(f"u_net_l{l}", u_net)
            u_tilde = F.relu(u + u_net) + 1e-8
            u_new = _safe_row_normalize(u_tilde, dim=1)
            if self.freeze_utilities:
                u_new = u
            _assert_finite(f"r_new_l{l}", r_new)
            _assert_finite(f"S_new_l{l}", S_new)
            _assert_finite(f"u_new_l{l}", u_new)

            new_states.append((r_new, S_new, u_new))

        new_states = self._enforce_global_resource_conservation(
            [self.get_state(l) for l in range(self.n_layers)],
            new_states,
        )

        # 3. Calcul des utilités AVANT de détacher les états (pour conserver le grad_fn)
        utilities = self._compute_utilities_from_states(new_states, p_r_up=list(P_R_up))

        # 4. Application des nouveaux états
        for l in range(self.n_layers):
            self.set_state(l, *new_states[l])

        return utilities

    def _compute_utilities_from_states(self, states, p_r_up=None):
        """
        Calcule la fonction d'utilité Cobb-Douglas à partir d'une liste
        de (r, S, u) non détachés.

        p_r_up : liste de tenseurs P_up par couche (shape N_l × N_{l+1} × n_res),
                 requis pour utility_mode='aggregate'.
        """
        utilities = []
        for l, (r, _, u) in enumerate(states):
            _assert_finite(f"utility_r_l{l}", r)
            _assert_finite(f"utility_u_l{l}", u)
            u = _safe_row_normalize(u, dim=1)

            if self.utility_mode == 'aggregate' and p_r_up is not None and l > 0:
                # α[k,i,m] = P_up[l-1][k,i,m] / Σ_j P_up[l-1][k,j,m]
                p = p_r_up[l - 1]                              # (N_{l-1}, N_l, n_res)
                denom = p.sum(dim=1, keepdim=True).clamp(min=1e-8)
                alpha = p / denom                              # (N_{l-1}, N_l, n_res)
                r_children = states[l - 1][0]                 # (N_{l-1}, n_res)
                r_contrib = torch.einsum('kim, km -> im', alpha, r_children)
                r_eval = r + r_contrib
                _assert_finite(f"r_eval_l{l}", r_eval)
            else:
                r_eval = r

            if self.strict_cobb_douglas:
                log_r = torch.log(r_eval.clamp(min=1e-8))
            else:
                log_r = torch.log1p(r_eval.clamp(min=0.0))
            # Cas instable observé : free + leontief limité aux atomes peut produire
            # des utilités extrêmes et envoyer les logits d'allocation en NaN.
            if (
                self.resource_mode == 'free'
                and self.transform_mode == 'leontief'
                and self.atoms_only_transform
            ):
                log_r = log_r.clamp(max=30.0)

            _assert_finite(f"log_r_l{l}", log_r)
            log_U = torch.sum(u * log_r, dim=1)
            _assert_finite(f"log_U_l{l}", log_U)
            utilities.append(log_U)
        return utilities

    def compute_utilities(self):
        """
        Calcule la fonction d'utilité depuis les états buffers courants (sans grad).
        """
        states = [self.get_state(l) for l in range(self.n_layers)]
        p_r_up = [self.get_proportions(l, 'R')[0] for l in range(self.n_layers)]
        return self._compute_utilities_from_states(states, p_r_up=p_r_up)

    def aggregate_system_utility(self, utilities):
        """
        Agrège l'utilité par moyenne de couche pour ne pas favoriser mécaniquement
        les couches contenant davantage de nœuds.
        """
        return sum(u.mean() for u in utilities)

    def update_edge_memory(self, states):
        """
        [Reciprocity] Met à jour les EMA de mémoire d'arête pondérée par l'utilité.

        La mémoire M[i,j,m] trace la valeur utilitaire des ressources reçues de j :
            M[i,j,m] ← α·M[i,j,m] + (1-α)·u_i[m]·r_reçu_de_j[m]

        Ainsi le nœud i favorise les partenaires j qui lui envoient ce qu'il valorise,
        et délaisse ceux qui envoient des ressources peu utiles (Tit-for-Tat sélectif).
        """
        P_R_up, P_R_down = zip(*[self.get_proportions(l, 'R') for l in range(self.n_layers)])

        for l in range(self.n_layers):
            u_i = getattr(self, f'u_{l}').detach()  # (N_l, n_res)

            if l < self.n_layers - 1:
                r_next = states[l + 1][0]
                received_up = torch.einsum('jin, jn -> jin',
                                           P_R_down[l + 1].detach(), r_next.detach())
                received_up = received_up.permute(1, 0, 2)          # (N_l, N_{l+1}, n_res)
                weighted_up = received_up * u_i.unsqueeze(1)        # pondération utilité
                M = getattr(self, f'M_up_{l}')
                new_M_up = self.alpha * M + (1 - self.alpha) * weighted_up
                _assert_finite(f"M_up_{l}", new_M_up)
                setattr(self, f'M_up_{l}', new_M_up.detach())

            if l > 0:
                r_prev = states[l - 1][0]
                received_dn = torch.einsum('kin, kn -> kin',
                                           P_R_up[l - 1].detach(), r_prev.detach())
                received_dn = received_dn.permute(1, 0, 2)          # (N_l, N_{l-1}, n_res)
                weighted_dn = received_dn * u_i.unsqueeze(1)        # pondération utilité
                M = getattr(self, f'M_down_{l}')
                new_M_down = self.alpha * M + (1 - self.alpha) * weighted_dn
                _assert_finite(f"M_down_{l}", new_M_down)
                setattr(self, f'M_down_{l}', new_M_down.detach())

    def compute_reciprocity_loss(self):
        """
        [Reciprocity] Bonus différentiable : récompenser d'envoyer vers les partenaires
        qui ont historiquement renvoyé.
        Retourne un scalaire négatif (à minimiser pour maximiser le bonus).
        """
        P_R_up, P_R_down = zip(*[self.get_proportions(l, 'R') for l in range(self.n_layers)])
        bonus = torch.zeros((), dtype=self.logits_keep_R[0].dtype, device=self.logits_keep_R[0].device)

        for l in range(self.n_layers):
            r_sender, _, _ = self.get_state(l)
            sender_value = r_sender.mean(dim=1)
            # Cas instable observé : free + leontief limité aux atomes peut garder
            # des ressources finies mais trop grandes pour la somme du bonus.
            if (
                self.resource_mode == 'free'
                and self.transform_mode == 'leontief'
                and self.atoms_only_transform
            ):
                sender_value = sender_value.clamp(max=torch.exp(torch.tensor(30.0, dtype=sender_value.dtype, device=sender_value.device)))

            if l < self.n_layers - 1 and hasattr(self, f'M_up_{l}'):
                # Somme sur m (valeurs positives pondérées utilité) → score par arête
                M_score = getattr(self, f'M_up_{l}').sum(dim=2)
                _assert_finite(f"M_score_up_l{l}", M_score)
                M_pref = _safe_row_normalize(M_score, dim=1)
                P_mean = P_R_up[l].mean(dim=2)
                alignment = (M_pref * P_mean).sum(dim=1)
                bonus = bonus + (sender_value * M_pref.size(1) * alignment).sum()
            if l > 0 and hasattr(self, f'M_down_{l}'):
                M_score = getattr(self, f'M_down_{l}').sum(dim=2)
                _assert_finite(f"M_score_down_l{l}", M_score)
                M_pref = _safe_row_normalize(M_score, dim=1)
                P_mean = P_R_down[l].mean(dim=2)
                alignment = (M_pref * P_mean).sum(dim=1)
                bonus = bonus + (sender_value * M_pref.size(1) * alignment).sum()

        rec_loss = -self.lam * bonus
        _assert_finite("reciprocity_loss", rec_loss)
        return rec_loss

    def compute_market_loss(self):
        """
        [Market] Terme d'alignement offre/demande différentiable (Option A).

        Récompense l'envoi vers des partenaires dont les ressources correspondent
        aux besoins de l'émetteur — fournit un gradient pour augmenter les flux
        vers les partenaires alignés (symétrique au bonus de réciprocité).

          market_up   = Σ_l Σ_i Σ_j Σ_m  P_up[l][i,j,m]   · u_i[m]      · r_{l+1}[j,m]
          market_down = Σ_l Σ_i Σ_j Σ_m  P_down[l][i,j,m] · u_{l-1}[j,m] · r_l[i,m]
          loss = −lam · (market_up + market_down)
        """
        bonuses_up, bonuses_down = self._compute_market_bonuses()
        alignment = torch.zeros((), dtype=self.logits_keep_R[0].dtype,
                                device=self.logits_keep_R[0].device)

        for l in range(self.n_layers):
            prop_up, prop_down = self.get_proportions(
                l, 'R',
                bonus_up=bonuses_up[l],
                bonus_down=bonuses_down[l],
            )
            u_i = getattr(self, f'u_{l}').detach()  # (N_l, n_res)
            r_l = getattr(self, f'r_{l}').detach()  # (N_l, n_res)

            if prop_up is not None:
                r_up = getattr(self, f'r_{l+1}').detach()  # (N_{l+1}, n_res)
                # Σ_ijm  P_up[i,j,m] · u_i[m] · r_up[j,m]
                alignment = alignment + torch.einsum('ijm, im, jm ->', prop_up, u_i, r_up)

            if prop_down is not None:
                u_dn = getattr(self, f'u_{l-1}').detach()  # (N_{l-1}, n_res)
                # Σ_ijm  P_down[i,j,m] · u_dn[j,m] · r_l[i,m]
                alignment = alignment + torch.einsum('ijm, jm, im ->', prop_down, u_dn, r_l)

        market_loss = -self.lam * alignment
        _assert_finite("market_loss", market_loss)
        return market_loss


def log_step(t, utilities, model, prev_utilities=None):
    """Log détaillé de l'état du système à chaque step."""
    sep = "-" * 60
    print(f"\n{'='*60}")
    print(f"  STEP {t}")
    print(f"{'='*60}")

    total_utility = sum(u.sum().item() for u in utilities)
    print(f"  Utilité totale du système : {total_utility:.4f}")

    if prev_utilities is not None:
        deltas = [
            (u - p).abs().mean().item()
            for u, p in zip(utilities, prev_utilities)
        ]
        print(f"  Delta moyen par couche   : {[f'{d:.4f}' for d in deltas]}")

    print(sep)
    for l, u in enumerate(utilities):
        u_np = u.detach()
        r, S, _ = model.get_state(l)

        prop_R_up, prop_R_down = model.get_proportions(l, 'R')
        keep_frac = 1.0
        if prop_R_up is not None:
            keep_frac -= prop_R_up.sum(dim=1).mean().item()
        if prop_R_down is not None:
            keep_frac -= prop_R_down.sum(dim=1).mean().item()

        print(
            f"  Couche {l} | "
            f"U: moy={u_np.mean():.4f}  std={u_np.std():.4f}  "
            f"[{u_np.min():.4f}, {u_np.max():.4f}] | "
            f"||r||: moy={r.norm(dim=1).mean():.4f} | "
            f"||S||_F: moy={S.norm(dim=(1,2)).mean():.4f} | "
            f"keep_R≈{keep_frac:.2%}"
        )
    print(sep)


import gzip
import numpy as np


def _pkl_dump(obj, path):
    """Sauvegarde atomique avec gzip (compresslevel=1 : rapide, ~3-5x réduction)."""
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wb', compresslevel=1) as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _pkl_load(path):
    """Chargement compatible gzip et pickle brut."""
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, 'rb') as f:
            return pickle.load(f)


def run_experiment(cfg: dict, verbose: bool = True) -> str:
    """
    Exécute un run complet avec la config donnée et retourne le run_id.

    Optimisations disque :
      - HIST_STRIDE : n'enregistre qu'1 step sur HIST_STRIDE (défaut 10 → 10x)
      - float32 au lieu de float64 (2x)
      - gzip compresslevel=1 sur tous les fichiers pkl (3-5x)
      - résultat attendu : ~40-80 Mo par run au lieu de plusieurs Go

    Reprise intra-run :
      - à chaque flush, un checkpoint (runs/{chash}_ckpt.pkl) sauvegarde l'état
        du modèle/optimiseur et le numéro de step atteint
      - au redémarrage, si le checkpoint existe, on repart de là
    """
    CHUNK_SIZE  = 2000   # steps réels entre deux flushes sur disque
    HIST_STRIDE = 10     # 1 snapshot gardé tous les HIST_STRIDE steps

    layer_sizes           = cfg['layer_sizes']
    n_res                 = cfg['n_res']
    n_steps               = cfg['n_steps']
    lr                    = cfg['lr']
    MODE                  = cfg['MODE']
    TRANSFORM_MODE        = cfg['TRANSFORM_MODE']
    N_REACTIONS           = cfg['N_REACTIONS']
    ATOMS_ONLY_TRANSFORM  = cfg['ATOMS_ONLY_TRANSFORM']
    UTILITY_MODE          = cfg['UTILITY_MODE']
    MARKET_STRENGTH       = cfg['MARKET_STRENGTH']
    BUDGET_MODE           = cfg['BUDGET_MODE']
    RESOURCE_MODE         = cfg['RESOURCE_MODE']
    STRICT_COBB_DOUGLAS   = cfg['STRICT_COBB_DOUGLAS']
    NO_PARENT_UTILITY     = cfg['NO_PARENT_UTILITY']
    NO_PARENT_SKILL       = cfg['NO_PARENT_SKILL']
    NO_RESOURCE_SELECTION = cfg['NO_RESOURCE_SELECTION']
    NO_SKILL_SELECTION    = cfg['NO_SKILL_SELECTION']
    NO_UTILITY_SELECTION  = cfg['NO_UTILITY_SELECTION']
    FREEZE_SKILLS         = cfg['FREEZE_SKILLS']
    FREEZE_UTILITIES      = cfg['FREEZE_UTILITIES']
    alpha_mem             = cfg['alpha_mem']
    lambda_rec            = cfg['lambda_rec']
    seed                  = cfg.get('seed', 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Config + hash ─────────────────────────────────────────────────────────
    _config = dict(
        layer_sizes=layer_sizes, n_res=n_res, n_steps=n_steps, lr=lr,
        MODE=MODE, TRANSFORM_MODE=TRANSFORM_MODE, N_REACTIONS=N_REACTIONS,
        ATOMS_ONLY_TRANSFORM=ATOMS_ONLY_TRANSFORM, UTILITY_MODE=UTILITY_MODE,
        MARKET_STRENGTH=MARKET_STRENGTH, BUDGET_MODE=BUDGET_MODE,
        RESOURCE_MODE=RESOURCE_MODE, STRICT_COBB_DOUGLAS=STRICT_COBB_DOUGLAS,
        NO_PARENT_UTILITY=NO_PARENT_UTILITY, NO_PARENT_SKILL=NO_PARENT_SKILL,
        NO_SKILL_SELECTION=NO_SKILL_SELECTION, NO_UTILITY_SELECTION=NO_UTILITY_SELECTION,
        FREEZE_SKILLS=FREEZE_SKILLS, FREEZE_UTILITIES=FREEZE_UTILITIES,
        alpha_mem=alpha_mem, lambda_rec=lambda_rec, seed=seed,
    )
    _chash    = hashlib.md5(str(sorted(_config.items())).encode()).hexdigest()[:6]
    os.makedirs('runs', exist_ok=True)
    _ckpt_path = os.path.join('runs', f'{_chash}_ckpt.pkl')

    # ── Instanciation ──────────────────────────────────────────────────────────
    model = EconomicHierarchicalDAG(
        layer_sizes, n_res,
        mode=MODE,
        alpha=alpha_mem,
        lam=lambda_rec,
        resource_mode=RESOURCE_MODE,
        no_parent_utility=NO_PARENT_UTILITY,
        no_parent_skill=NO_PARENT_SKILL,
        no_resource_selection=NO_RESOURCE_SELECTION,
        no_skill_selection=NO_SKILL_SELECTION,
        no_utility_selection=NO_UTILITY_SELECTION,
        freeze_skills=FREEZE_SKILLS,
        freeze_utilities=FREEZE_UTILITIES,
        strict_cobb_douglas=STRICT_COBB_DOUGLAS,
        transform_mode=TRANSFORM_MODE,
        n_reactions=N_REACTIONS,
        atoms_only_transform=ATOMS_ONLY_TRANSFORM,
        utility_mode=UTILITY_MODE,
        market_strength=MARKET_STRENGTH,
        budget_mode=BUDGET_MODE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Reprise depuis checkpoint si disponible ────────────────────────────────
    if os.path.exists(_ckpt_path):
        ckpt = _pkl_load(_ckpt_path)
        _run_id     = ckpt['run_id']
        _ts         = ckpt['timestamp']
        _start_t    = ckpt['t_completed'] + 1
        chunk_index = ckpt['chunk_index']
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        del ckpt
        gc.collect()
        if verbose:
            print(f"  [reprise] run_id={_run_id}  step={_start_t}  chunk={chunk_index}")
    else:
        _ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        _run_id  = f"{_ts}_{_chash}"
        _start_t = 0
        chunk_index = 0

    _run_path  = os.path.join('runs', f'{_run_id}.pkl')

    if verbose:
        print(f"Système : {len(layer_sizes)} couches, {layer_sizes[0]} nœuds/couche, {n_res} ressources")
        market_info = f"  market_strength={MARKET_STRENGTH}" if MODE == 'market' else ""
        print(f"Optimiseur : Adam  lr={lr}  steps={n_steps}  mode={MODE}{market_info}  transform={TRANSFORM_MODE}  atoms_only={ATOMS_ONLY_TRANSFORM}  utility={UTILITY_MODE}")
        print(f"Run ID  : {_run_id}   →  {_run_path}")
        print(f"Stockage : HIST_STRIDE={HIST_STRIDE}  CHUNK_SIZE={CHUNK_SIZE}  gzip compresslevel=1  float32")

    # ── Utilitaires de sauvegarde ──────────────────────────────────────────────
    def _save_async(snap, path):
        def _write():
            try:
                _pkl_dump(snap, path)
            except Exception as exc:
                print(f"  [warn] sauvegarde échouée ({path}): {exc}", flush=True)
        threading.Thread(target=_write, daemon=True).start()

    def _make_snap(n_done, hist_buffer):
        return {
            'run_id':      _run_id,
            'config':      _config,
            'layer_sizes': layer_sizes,
            'n_res':       n_res,
            'history': {
                'r':      list(hist_buffer['r']),
                'u':      list(hist_buffer['u']),
                'util':   list(hist_buffer['util']),
                'P_up':   list(hist_buffer['P_up']),
                'P_down': list(hist_buffer['P_down']),
                'steps':  list(hist_buffer['steps']),
            },
            'metadata': {
                'timestamp':         _ts,
                'n_steps_completed': n_done,
                'hist_stride':       HIST_STRIDE,
            },
        }

    def _flush_chunk(hist, idx, t_completed):
        chunk_path = os.path.join('runs', f'{_run_id}_chunk_{idx:04d}.pkl')
        _pkl_dump({'chunk_index': idx, 'history': {k: list(v) for k, v in hist.items()}}, chunk_path)
        # Checkpoint : état du modèle + position dans le run
        _pkl_dump({
            'run_id':          _run_id,
            'timestamp':       _ts,
            't_completed':     t_completed,
            'chunk_index':     idx + 1,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, _ckpt_path)
        return dict(r=[], u=[], util=[], P_up=[], P_down=[], steps=[])

    prev_utilities = None
    history = dict(r=[], u=[], util=[], P_up=[], P_down=[], steps=[])

    for t in range(_start_t, n_steps):

        # ══════════════════════════════════════════════════════════════════════
        # MODE : baseline
        # ══════════════════════════════════════════════════════════════════════
        if MODE == 'baseline':
            optimizer.zero_grad()
            utilities = model()
            loss = -model.aggregate_system_utility(utilities)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ══════════════════════════════════════════════════════════════════════
        # MODE : reciprocity  (Mémoire d'arête + bonus de réciprocité)
        # ══════════════════════════════════════════════════════════════════════
        elif MODE == 'reciprocity':
            optimizer.zero_grad()

            utilities = model()

            primary_loss = -model.aggregate_system_utility(utilities)
            rec_loss     = model.compute_reciprocity_loss()
            loss = primary_loss + rec_loss
            loss.backward()
            if (
                RESOURCE_MODE == 'free'
                and TRANSFORM_MODE == 'leontief'
                and ATOMS_ONLY_TRANSFORM
            ):
                _sanitize_nonfinite_grads(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Mise à jour des mémoires d'arête APRÈS le pas d'optimisation
            with torch.no_grad():
                post_states = [model.get_state(l) for l in range(model.n_layers)]
                model.update_edge_memory(post_states)

        # ══════════════════════════════════════════════════════════════════════
        # MODE : market  (Signaux offre/demande + alignement différentiable)
        # ══════════════════════════════════════════════════════════════════════
        elif MODE == 'market':
            optimizer.zero_grad()
            utilities = model()
            primary_loss = -model.aggregate_system_utility(utilities)
            market_loss  = model.compute_market_loss()
            loss = primary_loss + market_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        else:
            raise ValueError(f"MODE inconnu : {MODE!r}")

        # ══════════════════════════════════════════════════════════════════════
        # Collecte de l'historique (1 step sur HIST_STRIDE, en float32)
        # ══════════════════════════════════════════════════════════════════════
        if t % HIST_STRIDE == 0:
            with torch.no_grad():
                history['steps'].append(t)
                history['r'].append(
                    [model.get_state(l)[0].numpy().astype(np.float32) for l in range(model.n_layers)]
                )
                history['u'].append(
                    [model.get_state(l)[2].detach().numpy().astype(np.float32) for l in range(model.n_layers)]
                )
                history['util'].append(
                    [utilities[l].detach().numpy().astype(np.float32) for l in range(model.n_layers)]
                )
                p_up_snap, p_down_snap = [], []
                for l in range(model.n_layers):
                    pu, pd = model.get_proportions(l, 'R')
                    p_up_snap.append(pu.detach().numpy().astype(np.float32) if pu is not None else None)
                    p_down_snap.append(pd.detach().numpy().astype(np.float32) if pd is not None else None)
                history['P_up'].append(p_up_snap)
                history['P_down'].append(p_down_snap)

        if verbose and (t % 5 == 0 or t == n_steps - 1):
            with torch.no_grad():
                log_step(t, utilities, model, prev_utilities)

        prev_utilities = [u.detach().clone() for u in utilities]

        # ── Flush périodique de l'historique sur disque ────────────────────────
        if (t + 1) % CHUNK_SIZE == 0:
            history = _flush_chunk(history, chunk_index, t)
            chunk_index += 1
            gc.collect()

    # ── Flush du dernier buffer résiduel ──────────────────────────────────────
    if any(len(v) > 0 for v in history.values()):
        history = _flush_chunk(history, chunk_index, n_steps - 1)
        chunk_index += 1

    # ── Fusion des chunks → pkl final (gzip) ──────────────────────────────────
    if verbose:
        print("\nSimulation terminée. Fusion des chunks d'historique…")
    full_history = dict(r=[], u=[], util=[], P_up=[], P_down=[], steps=[])
    for i in range(chunk_index):
        chunk_path = os.path.join('runs', f'{_run_id}_chunk_{i:04d}.pkl')
        chunk = _pkl_load(chunk_path)
        for k in full_history:
            full_history[k].extend(chunk['history'].get(k, []))
        del chunk
        gc.collect()

    if verbose:
        print("Sauvegarde du run complet (gzip)…")
    _pkl_dump({
        'run_id':      _run_id,
        'config':      _config,
        'layer_sizes': layer_sizes,
        'n_res':       n_res,
        'history':     full_history,
        'metadata':    {
            'timestamp':         _ts,
            'n_steps_completed': n_steps,
            'hist_stride':       HIST_STRIDE,
        },
    }, _run_path)
    if verbose:
        print(f"Run sauvegardé : {_run_path}")

    _meta_path = _run_path.replace('.pkl', '.json')
    with open(_meta_path, 'w') as f:
        json.dump({
            'run_id':      _run_id,
            'timestamp':   _ts,
            'n_steps':     n_steps,
            'hist_stride': HIST_STRIDE,
            'layer_sizes': layer_sizes,
            'n_res':       n_res,
            'config':      _config,
        }, f, indent=2, default=str)

    # ── Suppression des chunks et du checkpoint ────────────────────────────────
    for i in range(chunk_index):
        chunk_path = os.path.join('runs', f'{_run_id}_chunk_{i:04d}.pkl')
        try:
            os.remove(chunk_path)
        except OSError:
            pass
    try:
        os.remove(_ckpt_path)
    except OSError:
        pass

    # ── Nettoyage RAM ──────────────────────────────────────────────────────────
    del model, optimizer, history, prev_utilities, full_history
    gc.collect()
    torch.cuda.empty_cache()

    return _run_id


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    cfg = dict(
        layer_sizes           = [10] * 10,
        n_res                 = 15,
        n_steps               = 20000,
        lr                    = 1e-3,
        MODE                  = 'reciprocity',
        TRANSFORM_MODE        = 'mass_action',
        N_REACTIONS           = 10,
        ATOMS_ONLY_TRANSFORM  = True,
        UTILITY_MODE          = 'self',
        MARKET_STRENGTH       = 1.0,
        BUDGET_MODE           = 'joint',
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
    run_experiment(cfg)
