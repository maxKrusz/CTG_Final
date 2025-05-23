import math
import os
import time

import torch

from attacks.popskipjump.abstract_attack import Attack
from attacks.popskipjump.defaultparams import DefaultParams
from attacks.popskipjump.infomax import get_n_from_cos, get_cos_from_n, bin_search
from attacks.popskipjump.tracker import InfoMaxStats

# Taken from https://github.com/cjsg/PopSkipJump

class PopSkipJump(Attack):
    def __init__(self, model_interface, params: DefaultParams = None):

        dataset = model_interface.models[0].dataset
        if dataset == 'cifar10':
            data_shape = 32, 32, 3
        if dataset == 'cifar100':
            data_shape = 32, 32, 3
        if dataset == 'imagenet':
            data_shape = 224, 224, 3

        super().__init__(model_interface, data_shape, torch.device('cuda'), params)
        self.theta_prob = 1. / self.grid_size  # Theta for Info-max procedure
        if self.constraint == 'l2':
            self.delta_det_unit = self.theta_det * math.sqrt(self.d)
            self.delta_prob_unit = math.sqrt(self.d) / self.grid_size  # PSJA's delta in unit scale
        elif self.constraint == 'linf':
            self.delta_det_unit = self.theta_det * self.d
            self.delta_prob_unit = self.d / self.grid_size  # PSJA's delta in unit scale
        self.stop_criteria = params.infomax_stop_criteria

    def bin_search_step(self, original, perturbed, page=None, estimates=None, step=None):
        if self.targeted:
            label = self.a.targeted_label
        else:
            label = self.a.true_label
        perturbed, dist_post_update, s_, e_, t_, n_, (nn_tmap, xx) = self.info_max_batch(
            original, perturbed[None], label, estimates, step)
        if page is not None:
            page.info_max_stats = InfoMaxStats(s_, t_, xx, e_, n_)
        return perturbed, dist_post_update, {'s': s_, 'e': e_, 'n': n_, 't': t_}

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        if self.constraint == "l2":
            delta_prob_unit = self.theta_prob * math.sqrt(self.d)  # PSJA's delta in unit scale
        elif self.constraint == "linf":
            delta_prob_unit = self.theta_prob * self.d  # PSJA's delta in unit scale
        delta_prob = dist_post_update * delta_prob_unit  # PSJA's delta

        num_evals_prob = estimates['n']
        page.num_eval_prob = num_evals_prob
        num_evals_prob = int(min(num_evals_prob, self.max_num_evals))
        page.time.num_evals = time.time()
        return self._gradient_estimator(perturbed, num_evals_prob, delta_prob)

    def make_gradient_step(self, epsilon, perturbed, update):
        if self.constraint == 'l2':
            perturbed = torch.clamp(perturbed + epsilon * update, self.clip_min, self.clip_max)
        elif self.constraint == 'linf':
            perturbed = torch.clamp(perturbed + epsilon * update, self.clip_min, self.clip_max)
            # if len(self.diary.iterations) == 0:
            #     x_tminus1 = self.diary.initial_projection
            # else:
            #     x_tminus1 = self.diary.iterations[-1].bin_search
            # x_tilde = perturbed
            # mid = 0.5 * (x_tilde + x_tminus1)
            # perturbed = torch.clamp(mid + epsilon * update, self.clip_min, self.clip_max)
        else:
            raise RuntimeError(f"Unknown constraint: {self.constraint}")
        return perturbed

    def opposite_movement_step(self, original, perturbed):
        # Go in the opposite direction
        if self.constraint == 'l2':
            return torch.clamp(perturbed + 0.5 * (perturbed - original), self.clip_min, self.clip_max)
        elif self.constraint == 'linf':
            eps = torch.max(torch.abs(original - perturbed))
            diff = perturbed - original
            if self.step < 15:
                alpha = 1
            elif self.step < 30:
                alpha = 3
            else:
                alpha = 5
            # alpha = 1  # alpha = 1 for l2 and alpha = infinity for linf
            perturbed = perturbed + 0.5 * eps * (diff / eps) ** alpha
            return perturbed
            # return torch.clamp(perturbed + 0.5 * (perturbed - original), self.clip_min, self.clip_max)
            # o_ = original.flatten()
            # p_ = perturbed.flatten()
            # indices = torch.argmax(torch.abs(o_ - p_))
            # p_[indices] = p_[indices] + 0.5 * (p_[indices] - o_[indices])
            # return torch.clamp(p_.view(perturbed.shape), self.clip_min, self.clip_max)

    def get_theta_prob(self, target_cos, estimates=None):
        """
        Performs binary search for finding maximal theta_prob that does not affect estimated samples
        """
        # TODO: Replace Binary Search with a closed form solution (if it exists)
        if estimates is None:
            s, eps = 100., 1e-4
            n1 = get_n_from_cos(target_cos, s=s, theta=0, delta=self.delta_prob_unit, d=self.d, eps=eps)
        else:
            s, eps = estimates['s'], estimates['e']
            n1 = get_n_from_cos(target_cos, s=s, theta=0, delta=self.delta_prob_unit, d=self.d, eps=eps)
        low, high = 0, self.theta_det
        theta = self.theta_det
        n2 = get_n_from_cos(target_cos, s=s, theta=theta, delta=self.delta_prob_unit, d=self.d, eps=eps)
        while (n2 - n1) < 1:
            theta *= 2
            n2 = get_n_from_cos(target_cos, s=s, theta=theta, delta=self.delta_prob_unit, d=self.d, eps=eps)
            low, high = theta / 2, theta

        while high - low >= self.theta_det:
            mid = (low + high) / 2
            n2 = get_n_from_cos(target_cos, s=s, theta=mid, delta=self.delta_prob_unit, d=self.d, eps=eps)
            if (n2 - n1) < 1:
                low = mid
            else:
                high = mid
        return low

    def info_max_batch(self, unperturbed, perturbed_inputs, label, estimates, step):

        if self.prior_frac == 0:
            if step is None or step <= 1:
                prior_frac = 1
            elif step <= 4:
                prior_frac = 0.5
            elif step <= 10:
                prior_frac = 0.2
            else:
                prior_frac = 0.1
        else:
            prior_frac = self.prior_frac
        border_points = []
        dists = []
        smaps, tmaps, emaps, ns = [], [], [], []
        if estimates is None:
            target_cos = get_cos_from_n(self.initial_num_evals, theta=self.theta_det, delta=self.delta_det_unit,
                                        d=self.d)
        else:
            num_evals_det = int(min([self.initial_num_evals * math.sqrt(step + 1), self.max_num_evals]))
            target_cos = get_cos_from_n(num_evals_det, theta=self.theta_det, delta=self.delta_det_unit, d=self.d)
        # theta_prob_dynamic = self.get_theta_prob(target_cos, estimates)
        # grid_size_dynamic = min(self.grid_size, int(1 / theta_prob_dynamic) + 1)
        grid_size_dynamic = self.grid_size
        for perturbed_input in perturbed_inputs:
            output, n = bin_search(
                unperturbed, perturbed_input, self.model_interface, d=self.d,
                grid_size=grid_size_dynamic, device=self.device, delta=self.delta_prob_unit,
                label=label, targeted=self.targeted, prev_t=self.prev_t, prev_s=self.prev_s,
                prev_e=self.prev_e, prior_frac=prior_frac, target_cos=target_cos,
                queries=self.queries, plot=False, stop_criteria=self.stop_criteria, dist_metric=self.constraint)
            nn_tmap_est = output['nn_tmap_est']
            t_map, s_map, e_map = output['ttse_max'][-1]
            num_retries = 0
            while t_map == 1 and num_retries < 5:
                num_retries += 1
                # print(f'Got t_map == 1, Retrying {num_retries}...')
                output, n = bin_search(
                    unperturbed, perturbed_input, self.model_interface, d=self.d,
                    grid_size=grid_size_dynamic, device=self.device, delta=self.delta_prob_unit,
                    label=label, targeted=self.targeted, prev_t=self.prev_t, prev_s=self.prev_s,
                    prev_e=self.prev_e, prior_frac=prior_frac, target_cos=target_cos,
                    queries=self.queries, plot=False, stop_criteria=self.stop_criteria, dist_metric=self.constraint)
                nn_tmap_est = output['nn_tmap_est']
                t_map, s_map, e_map = output['ttse_max'][-1]
            if t_map == 1:
                # print('Prob of label (unperturbed):', self.model_interface.get_probs(unperturbed)[0, label])
                # print('Prob of label (perturbed):', self.model_interface.get_probs(perturbed_input)[0, label])
                space = [(1 - tt) * perturbed_input + tt * unperturbed for tt in torch.linspace(0, 1, 21)]
                # print([self.model_interface.get_probs(x)[0, label] for x in space])
                # print('delta:', self.delta_prob_unit)
                # print('label:', label)
                # print('prev_t,s,e:', self.prev_t, self.prev_s, self.prev_e)
                # print('prior_frac:', prior_frac)
                # print('target_cos', target_cos)

                if not os.path.exists(f"{os.path.dirname(__file__)}/dumps"):
                    os.makedirs(f"{os.path.dirname(__file__)}/dumps")

                torch.save(unperturbed, open(f'{os.path.dirname(__file__)}/dumps/unperturbed.pkl', 'wb'))
                torch.save(perturbed_input, open(f'{os.path.dirname(__file__)}/dumps/perturbed.pkl', 'wb'))
                t_map = 1.0 - 0.5 / self.grid_size
                # torch.save(self.model_interface, open('dumps/model_interface.pkl', 'wb'))

            if self.constraint == 'l2':
                border_point = (1 - t_map) * perturbed_input + t_map * unperturbed
            elif self.constraint == 'linf':
                dist_linf = self.compute_distance(unperturbed, perturbed_input)
                alphas = (1 - t_map) * dist_linf
                border_point = self.project(unperturbed, perturbed_input, alphas[None])[0]
            self.prev_t, self.prev_s, self.prev_e = t_map, s_map, e_map
            dist = self.compute_distance(unperturbed, border_point)
            border_points.append(border_point)
            dists.append(dist)
            smaps.append(s_map)
            tmaps.append(t_map)
            emaps.append(e_map)
            ns.append(n)
        idx = int(torch.argmin(torch.tensor(dists)))
        dist = self.compute_distance(unperturbed, perturbed_inputs[idx])
        if dist == 0:
            print("Distance is zero in search")
        out = border_points[idx]
        dist_border = self.compute_distance(out, unperturbed)
        if dist_border == 0:
            print("Distance of border point is 0")
        return out, dist, smaps[idx], emaps[idx], tmaps[idx], ns[idx], (nn_tmap_est, output['xxj'])

    def geometric_progression_for_stepsize(self, x, update, dist, current_iteration, original=None):
        return dist / math.sqrt(current_iteration)
