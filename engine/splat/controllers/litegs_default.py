"""LiteGS default density controller (TamingGS scores) with a real Gaussian budget."""

from __future__ import annotations

from engine.splat.controllers import DensifySchedule


class LiteGSController:
    """Thin wrapper over ``DensityControllerTamingGS``: same maths, plus event lines
    and a state dict so Continue restores the schedule. ``epoch`` arguments are cycles.

    Vendor TamingGS ``prune_mode='weight'`` deletes ``fragment_weight==0`` Gaussians.
    That is correct when the stats window is a full epoch (LiteGS's epoch-aligned
    schedule). On a shorter cycle it would delete Gaussians that only unvisited
    cameras can see -- ``_guard_unseen_prune`` then keeps an opacity-only cull.
    """

    name = "LiteGS"
    needs_mean2d_grad = False

    def __init__(self, backend, budget: int, schedule: DensifySchedule, log=None) -> None:
        self.backend = backend
        self.log = log
        self.schedule = schedule
        self.events: list[str] = []
        dp = backend.dp
        dp.target_primitives = int(max(budget, 1))
        dp.opacity_reset_mode = "decay"
        dp.prune_mode = "weight"
        schedule.apply(dp)
        init_points = int(backend.init_points)
        self.inner = backend.lg_densify.DensityControllerTamingGS(
            backend.norm_radius, dp, backend.pp.cluster_size > 0, init_points
        )
        self.budget = int(dp.target_primitives)
        n_train = int(len(getattr(backend, "train_views", ()) or ()))
        # Vendor TamingGS weight-prune is valid only when the stats window is a full
        # epoch of every training view. Epoch-aligned schedules (interval_steps >= n_train)
        # match that; shorter cycles still need the opacity-only guard.
        if schedule.interval_steps < max(n_train, 1):
            self._guard_unseen_prune()
            prune_note = "prune: transparent only (cycle shorter than one epoch)"
        else:
            prune_note = "prune: vendor weight==0 (full-epoch stats)"
        self.events.append(f"LiteGS TamingGS: budget {self.budget}; {schedule.describe()} ({prune_note})")

    def _guard_unseen_prune(self) -> None:
        """Don't treat 'not rasterised this cycle' as 'never visible'."""
        min_opacity = float(self.inner.min_opacity)

        def get_prune_mask(actived_opacity, actived_scale):
            del actived_scale
            return actived_opacity.reshape(-1) < min_opacity

        self.inner.get_prune_mask = get_prune_mask

    def is_densify_active(self, cycle: int) -> bool:
        return bool(self.inner.is_densify_actived(cycle))

    def step(self, optimizer, cycle: int):
        before = self.backend.n_gaussians
        self.backend.ensure_optimizer_state()
        out = self.inner.step(optimizer, cycle)
        after = self.backend.n_gaussians
        dp = self.inner.densify_params
        if dp.densify_from <= cycle < dp.densify_until:
            bits = [f"densify {before} -> {after}"]
            if cycle % dp.opacity_reset_interval == 0:
                bits.append("opacity decay x0.5")
            self.events.append(f"cycle {cycle}: " + ", ".join(bits) + f" (budget {self.budget})")
        return out

    def state_dict(self) -> dict:
        return {"kind": self.name, "budget": self.budget}

    def load_state_dict(self, state: dict) -> None:
        return


def build(backend, *, budget: int, schedule: DensifySchedule, log=None) -> LiteGSController:
    return LiteGSController(backend, budget, schedule, log=log)
