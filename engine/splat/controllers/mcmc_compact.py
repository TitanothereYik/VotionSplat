"""MCMC Compact density controller — clean-room Jung & Hong 2025, section 3.1.

Jung, Hong: *MCMC-Guided Compact 3D Gaussian Splatting for Relightable Rendering*,
SIGGRAPH Asia 2025 Technical Communications (doi 10.1145/3757376.3771401). No public
code; only the non-relighting part (3.1) is reproduced:

1. **Adaptive split threshold by Metropolis-Hastings.** Instead of the fixed 3DGS
   gradient threshold, ``eps_curr`` is a Markov chain over log-threshold whose target
   density is ``exp(-(split_ratio(eps) - delta_split)^2 / (2 sigma_r^2))`` where
   ``split_ratio(eps)`` is the fraction of Gaussians with mean 2-D positional gradient
   ``>= eps``. Proposals are log-normal; the proposal width is tuned toward the ~0.44
   optimal acceptance rate (Roberts & Rosenthal). So each densify splits about
   ``delta_split`` of the Gaussians regardless of scene scale or gradient units.
2. **Stochastic importance prune (Mini-Splatting style).** After split/clone, when the
   count exceeds the budget, keep ``budget`` Gaussians sampled *without replacement*
   with probability proportional to their accumulated blend weight (LiteGS
   ``fragment_weight`` statistic). Transparent (``opacity < 0.005``) and never-visible
   Gaussians are always dropped first.

The normal-gradient term of the paper is omitted (Votion learns no normals). The MALA
light sampling, BRDF and deferred shading are relighting features and out of scope.

Runs on the LiteGS optimizer with LiteGS layouts (``[..., chunks, chunk_size]``). LiteGS
clusters Gaussians in chunks of ``CLUSTER_SIZE`` (128), so every append / prune here is
**chunk-exact**: appends are topped up to the next chunk with the next-best gradient
candidates (LiteGS would drop a sub-chunk append; that stalls growth below ~20k
Gaussians), prunes remove a chunk multiple (``cluster_points`` would otherwise pad the
remainder with *duplicated* Gaussians). Once the live count is at the budget, split/clone
stops so the importance resample cannot re-compete the whole population every cycle
(that churn was the 80 %-window PSNR stall; ending densify at 40 % was a workaround that
starved short 10k runs at ~117k Gaussians).

Cadence: the 3DGS ``densification_interval`` (``defaults.MCMC_DENSIFY_INTERVAL`` = 100
steps) over the same 5 %..80 % window and ~2000-step opacity-decay spacing as LiteGS. A
cycle then sees only part of the training views, so "never visible" is only trusted as a
prune reason when a cycle covers a whole epoch; otherwise transparency and the budget
sampler do the pruning.
"""

from __future__ import annotations

import math

from engine.splat import defaults as D
from engine.splat.controllers import DensifySchedule


class MCMCCompactController:
    name = "MCMC Compact"
    needs_mean2d_grad = True

    def __init__(
        self,
        backend,
        budget: int,
        schedule: DensifySchedule,
        *,
        delta_split: float = D.MCMC_DELTA_SPLIT,
        sigma_ratio: float = 0.01,
        mh_iterations: int = 48,
        target_accept: float = 0.44,
        log=None,
    ) -> None:
        self.backend = backend
        self.torch = backend.torch
        self.log = log
        self.events: list[str] = []
        self.n_train = int(len(getattr(backend, "train_views", ()) or ()))
        # 3DGS cadence over the same window; the trainer reads this back from .schedule.
        max_steps = int(getattr(backend.op, "iterations", 0) or 0)  # set by backend.configure_schedule
        if max_steps <= 0:
            max_steps = int(round(schedule.until_cycle * schedule.interval_steps / 0.8))
        schedule = DensifySchedule.for_run(
            max_steps,
            self.n_train,
            interval_steps=D.MCMC_DENSIFY_INTERVAL,
            until_fraction=D.MCMC_DENSIFY_UNTIL_FRACTION,
        )
        self.schedule = schedule
        self.full_coverage = schedule.interval_steps >= max(self.n_train, 1)
        dp = backend.dp
        dp.opacity_reset_mode = "decay"
        schedule.apply(dp)
        self.dp = dp
        self.budget = int(max(budget, 1))
        self.delta_split = float(delta_split)
        self.sigma_ratio = float(sigma_ratio)
        self.mh_iterations = int(mh_iterations)
        self.target_accept = float(target_accept)
        # MH chain state (persisted in the checkpoint)
        self.eps_curr = float(dp.densify_grad_threshold)  # 3DGS default as the chain's origin
        self.log_step = 0.5  # std of the log-normal proposal
        self.accepts = 0
        self.proposals = 0
        self.min_opacity = float(dp.opacity_threshold)
        self.tau_s = float(dp.percent_dense) * float(backend.norm_radius)
        self.b_cluster = backend.pp.cluster_size > 0
        self.inner = backend.lg_densify.DensityControllerOfficial(
            backend.norm_radius, dp, self.b_cluster, int(backend.init_points)
        )
        self.gen = self.torch.Generator(device="cpu")
        self.gen.manual_seed(20250907)
        self.events.append(
            f"MCMC Compact: budget {self.budget} delta_split {self.delta_split:.3f} "
            f"tau_s {self.tau_s:.4f}; {schedule.describe()}"
        )

    # ------------------------------------------------------------------ schedule
    def is_densify_active(self, cycle: int) -> bool:
        return bool(self.inner.is_densify_actived(cycle))

    def state_dict(self) -> dict:
        return {
            "kind": self.name,
            "budget": self.budget,
            "eps_curr": self.eps_curr,
            "log_step": self.log_step,
            "accepts": self.accepts,
            "proposals": self.proposals,
        }

    def load_state_dict(self, state: dict) -> None:
        try:
            self.eps_curr = float(state.get("eps_curr", self.eps_curr))
            self.log_step = float(state.get("log_step", self.log_step))
            self.accepts = int(state.get("accepts", 0))
            self.proposals = int(state.get("proposals", 0))
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------ MH threshold
    def _split_ratio(self, grads, eps: float) -> float:
        return float((grads >= eps).float().mean())

    def _log_target(self, ratio: float) -> float:
        d = ratio - self.delta_split
        return -(d * d) / (2.0 * self.sigma_ratio * self.sigma_ratio)

    def _walk_threshold(self, grads) -> tuple[float, float]:
        """Metropolis-Hastings on log(eps). Returns (eps, split_ratio at eps)."""
        torch = self.torch
        positive = grads[grads > 0]
        if positive.numel() == 0:
            return self.eps_curr, 0.0
        eps = float(self.eps_curr)
        lo = float(positive.min()) * 0.5
        hi = float(positive.max()) * 2.0
        eps = min(max(eps, lo), hi)
        ratio = self._split_ratio(grads, eps)
        logp = self._log_target(ratio)
        acc = 0
        noise = torch.randn(self.mh_iterations, generator=self.gen).tolist()
        unif = torch.rand(self.mh_iterations, generator=self.gen).tolist()
        for k in range(self.mh_iterations):
            prop = eps * math.exp(noise[k] * self.log_step)
            prop = min(max(prop, lo), hi)
            r_prop = self._split_ratio(grads, prop)
            lp_prop = self._log_target(r_prop)
            if math.log(max(unif[k], 1e-12)) < (lp_prop - logp):
                eps, ratio, logp = prop, r_prop, lp_prop
                acc += 1
        self.accepts += acc
        self.proposals += self.mh_iterations
        rate = acc / max(self.mh_iterations, 1)
        # Robbins-Monro tuning of the proposal width toward the 0.44 optimum.
        self.log_step = float(min(max(self.log_step * math.exp(0.6 * (rate - self.target_accept)), 0.02), 3.0))
        self.eps_curr = float(eps)
        return eps, ratio

    # --------------------------------------------------------------- densify ops
    def _params(self, optimizer):
        xyz, scale, rot, sh0, shr, opa = self.inner._get_params_from_optimizer(optimizer)
        chunk = xyz.shape[-1] if self.b_cluster else 0
        if self.b_cluster:
            xyz, scale, rot, sh0, shr, opa = self.backend.lg_scene.cluster.uncluster(xyz, scale, rot, sh0, shr, opa)
        return chunk, (xyz, scale, rot, sh0, shr, opa)

    def split_and_clone(self, optimizer) -> tuple[int, int, float, float]:
        torch = self.torch
        stats = self.backend.stats
        chunk, (xyz, scale, rot, sh0, shr, opa) = self._params(optimizer)
        n = xyz.shape[-1]
        got = stats.get_mean("mean2d_grad")
        if got is None:
            return 0, 0, self.eps_curr, 0.0
        grads, _count = got
        grads = grads.reshape(-1)[:n].nan_to_num(0.0)
        eps, ratio = self._walk_threshold(grads)
        candidates = grads >= eps
        n_cand = int(candidates.sum())
        if n_cand == 0:
            return 0, 0, eps, ratio
        if self.b_cluster and chunk > 0 and n_cand % chunk != 0:
            # LiteGS stores chunks of `chunk`; top the append up to the next chunk with the
            # next-best gradients instead of dropping the remainder (which stalls growth).
            want = min(-(-n_cand // chunk) * chunk, n)
            extra = want - n_cand
            if extra > 0:
                rest = torch.where(candidates, torch.full_like(grads, -1.0), grads)
                top = torch.topk(rest, extra, largest=True)
                more = top.indices[top.values > 0]
                candidates[more] = True
        big = scale.exp().max(dim=0).values > self.tau_s
        split_mask = candidates & big
        clone_mask = candidates & ~big
        n_split = int(split_mask.sum())
        n_clone = int(clone_mask.sum())
        if n_split + n_clone == 0:
            return 0, 0, eps, ratio
        split_idx = split_mask.nonzero()[:, 0]
        clone_idx = clone_mask.nonzero()[:, 0]
        if self.b_cluster and chunk > 0:
            # chunk-exact: the top-up above normally makes this a no-op; if the gradients
            # ran out, drop the tail (clones first) rather than let LiteGS pad duplicates.
            limit = ((n_split + n_clone) // chunk) * chunk
            if limit <= 0:
                return 0, 0, eps, ratio
            if limit < n_split + n_clone:
                n_split = min(n_split, limit)
                n_clone = limit - n_split
                split_idx = split_idx[:n_split]
                clone_idx = clone_idx[:n_clone]

        W = self.backend.lg_utils.wrapper
        # split: original shrinks to 1/1.6 and one sibling is sampled inside the old ellipsoid
        stds = scale[..., split_idx].exp()
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds).unsqueeze(0)
        tm = W.CreateTransformMatrix.call_fused(
            torch.ones_like(stds), torch.nn.functional.normalize(rot[..., split_idx], dim=0)
        )[:3, :3]
        shift = (samples.permute(2, 0, 1) @ tm.permute(2, 0, 1)).permute(1, 2, 0).squeeze(0)
        new_scale = (stds / 1.6).log()
        split_xyz = xyz[..., split_idx] + shift
        scale.data[..., split_idx] = new_scale  # in-place on the live parameter

        append = [
            torch.cat((split_xyz, xyz[..., clone_idx]), dim=-1),
            torch.cat((new_scale, scale[..., clone_idx]), dim=-1),
            torch.cat((rot[..., split_idx], rot[..., clone_idx]), dim=-1),
            torch.cat((sh0[..., split_idx], sh0[..., clone_idx]), dim=-1),
            torch.cat((shr[..., split_idx], shr[..., clone_idx]), dim=-1),
            torch.cat((opa[..., split_idx], opa[..., clone_idx]), dim=-1),
        ]
        if self.b_cluster:
            append = list(self.backend.lg_scene.cluster.cluster_points(chunk, *append))
        names = ["xyz", "scale", "rot", "sh_0", "sh_rest", "opacity"]
        self.inner._cat_tensors_to_optimizer(dict(zip(names, append)), optimizer)
        return n_split, n_clone, eps, ratio

    def prune(self, optimizer) -> tuple[int, int]:
        """Returns (hard-pruned, importance-pruned)."""
        torch = self.torch
        stats = self.backend.stats
        chunk, (xyz, scale, rot, sh0, shr, opa) = self._params(optimizer)
        n = xyz.shape[-1]
        actived_opacity = opa.sigmoid().reshape(-1)
        transparent = actived_opacity < self.min_opacity
        unseen = torch.zeros_like(transparent)  # not rasterised by any view of this cycle
        try:
            culled = stats.get_global_culling().reshape(-1)
            m = min(culled.shape[0], n)
            unseen[:m] = culled[:m]
        except Exception:
            pass
        # "never visible" is a prune reason only when the cycle covered every training view
        hard = transparent | unseen if self.full_coverage else transparent
        n_hard = int(hard.sum())
        if n_hard >= int(0.8 * n):  # a whole scene cannot vanish in one step; keep the opaque ones
            keep_n = max(n - int(0.5 * n), 1)
            hard = torch.zeros_like(hard)
            hard[torch.topk(actived_opacity, n - keep_n, largest=False).indices] = True
            n_hard = int(hard.sum())

        remaining = n - n_hard
        n_soft = 0
        keep = ~hard
        # removal priority (lower = removed first) used to make the prune chunk-exact
        priority = actived_opacity.clone()
        priority[hard] = -1.0 + actived_opacity[hard]
        if remaining > self.budget:
            importance = torch.zeros(n, device=xyz.device, dtype=torch.float32)
            got = stats.get_mean("fragment_weight")
            if got is not None:
                fw, fc = got
                ws = (fw.reshape(-1) * fc.reshape(-1).float()).nan_to_num(0.0)
                m = min(ws.shape[0], n)
                importance[:m] = ws[:m].clamp_min(0.0)
            seen_pos = importance[(~unseen) & (~hard) & (importance > 0)]
            if not self.full_coverage and seen_pos.numel() > 0:
                # Gaussians outside this cycle's views have no blend weight yet: give them the
                # mean visible weight so the sampler neither favours nor discards them.
                importance[unseen & ~hard] = float(seen_pos.mean())
            importance = importance * actived_opacity
            importance[hard] = 0.0
            positive = int((importance > 0).sum())
            if positive <= self.budget:
                # not enough visible weight to sample: keep every positive plus the most opaque rest
                chosen = importance > 0
                short = self.budget - positive
                if short > 0:
                    rest = torch.where(chosen | hard, torch.zeros_like(actived_opacity), actived_opacity)
                    chosen[torch.topk(rest, short, largest=True).indices] = True
            else:
                idx = torch.multinomial(importance, self.budget, replacement=False)
                chosen = torch.zeros(n, dtype=torch.bool, device=xyz.device)
                chosen[idx] = True
            soft = (~hard) & (~chosen)
            n_soft = int(soft.sum())
            keep = chosen & ~hard
            priority[soft] = importance[soft]  # least important go first
        n_remove = n_hard + n_soft
        if n_remove <= 0 or int(keep.sum()) <= 0:
            return 0, 0
        if self.b_cluster and chunk > 0:
            # chunk-exact: LiteGS pads a non-multiple remainder with duplicated Gaussians,
            # so remove a chunk multiple, keeping the least-bad of the removal set.
            fit = (n_remove // chunk) * chunk
            if fit <= 0:
                return 0, 0
            if fit < n_remove:
                remove_idx = (~keep).nonzero()[:, 0]
                order = torch.argsort(priority[remove_idx])
                spare = remove_idx[order[fit:]]
                keep = keep.clone()
                keep[spare] = True
                spared_hard = int(hard[spare].sum())
                n_hard -= spared_hard
                n_soft -= int(spare.numel()) - spared_hard
        self.inner._prune_optimizer(keep, optimizer)
        return n_hard, n_soft

    # -------------------------------------------------------------------- step
    def step(self, optimizer, cycle: int):
        dp = self.dp
        with self.torch.no_grad():
            if dp.densify_from <= cycle < dp.densify_until:
                updated = False
                if cycle % dp.densification_interval == 0:
                    before = self.backend.n_gaussians
                    self.backend.ensure_optimizer_state()
                    # At the cap the paper's importance prune would resample 100% of
                    # the population every cycle (nothing settles). Skip split/clone
                    # so prune only drops transparent Gaussians and Adam can refine.
                    if before >= self.budget:
                        n_split, n_clone, eps, ratio = 0, 0, self.eps_curr, 0.0
                    else:
                        n_split, n_clone, eps, ratio = self.split_and_clone(optimizer)
                    n_hard, n_soft = self.prune(optimizer)
                    after = self.backend.n_gaussians
                    extra = " at-budget skip split" if before >= self.budget else ""
                    self.events.append(
                        f"cycle {cycle}: split {n_split} clone {n_clone}{extra} (ratio {ratio:.3f} "
                        f"eps {eps:.3e} step {self.log_step:.2f} acc {self._acc_rate():.2f}) "
                        f"prune hard {n_hard} importance {n_soft}; {before} -> {after} "
                        f"(budget {self.budget})"
                    )
                    updated = True
                if cycle % dp.opacity_reset_interval == 0:
                    self.inner.reset_opacity(optimizer, cycle)
                    self.events.append(f"cycle {cycle}: opacity decay x0.5")
                    updated = True
                if updated:
                    xyz = self.inner._get_params_from_optimizer(optimizer)[0]
                    self.backend.stats.reset(xyz.shape[-2], xyz.shape[-1], self.inner.is_densify_actived)
                    self.torch.cuda.empty_cache()
            return self.inner._get_params_from_optimizer(optimizer)

    def _acc_rate(self) -> float:
        return self.accepts / max(self.proposals, 1)


def build(backend, *, budget: int, schedule: DensifySchedule, log=None) -> MCMCCompactController:
    return MCMCCompactController(backend, budget, schedule, log=log)
