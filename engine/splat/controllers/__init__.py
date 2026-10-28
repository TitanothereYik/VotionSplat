"""Density controllers over the LiteGS optimizer. One backend, two controllers.

``LiteGS`` (default): ``litegs.training.densify.DensityControllerTamingGS`` as-is, with
``target_primitives`` = the UI Gaussian budget (the cap is real).
``MCMC Compact``: clean-room Jung & Hong 2025 section 3.1 (Metropolis-Hastings adaptive
split threshold + stochastic importance prune to the budget). Relighting parts of that
paper are out of scope.

Schedule
--------
LiteGS's vendor trainer densifies by *epoch* (from epoch 3, every 5, until 80 % of
epochs, opacity decay every 10) with statistics accumulated over a **full pass of every
training view**. Mapping that onto short step-cycles (``densification_interval=1`` every
100-1000 steps) was the adapter blur bug: TamingGS split samples isotropic Gaussians off
the surface faster than Adam can flatten them (synth plane: z std 0.42 vs vendor 0.06),
and ``fragment_weight==0`` prune treated "not seen this cycle" as "never visible". LiteGS
therefore uses one cycle = one epoch, vendor's interval of 5 when the run is long enough,
and 1 (every epoch) on short runs so growth can still reach the budget. MCMC Compact
keeps the 3DGS 100-step cadence (``defaults.MCMC_DENSIFY_INTERVAL``) and skips split at
the cap. The trainer reads the cadence back from ``controller.schedule``.

Both controllers return the same tuple from ``step(optimizer, cycle)`` as LiteGS
controllers and expose ``events`` (strings the trainer forwards as ``SPLAT\\tctl`` lines).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.splat import defaults as D

DECAY_MIN_STEPS = 2000  # LiteGS decays every 10 epochs of ~200 views; 3DGS resets every 3000 it
VENDOR_DENSIFY_EVERY = 5  # vendor litegs/training/trainer.py densification_interval
VENDOR_DENSIFY_FROM = 3
VENDOR_OPACITY_RESET = 10
# Below this many epochs, densify every epoch so a 10k-step SphereSfM run can still grow.
SHORT_RUN_EPOCHS = 20


@dataclass(frozen=True)
class DensifySchedule:
    interval_steps: int  # one "cycle"; statistics accumulate over it
    from_cycle: int  # first cycle whose end densifies
    until_cycle: int  # exclusive
    reset_every: int = 2  # opacity decay every N cycles
    densify_every: int = 1  # dp.densification_interval (vendor LiteGS: 5 epochs)

    @classmethod
    def for_run(
        cls,
        max_steps: int,
        n_train: int,
        *,
        interval_steps: int | None = None,
        until_fraction: float = 0.8,
    ) -> "DensifySchedule":
        """``interval_steps`` fixes the cycle length (MCMC Compact: 3DGS 100-step
        cadence). LiteGS default is one cycle = one epoch, matching vendor trainer.py.

        ``until_fraction`` is the 3DGS-standard 80 % for both controllers. MCMC Compact
        skips split/clone once N hits the budget (see ``mcmc_compact.step``).
        """
        max_steps = max(int(max_steps), 1)
        n_train = max(int(n_train), 1)
        until_frac = max(0.0, min(1.0, float(until_fraction)))
        if interval_steps is not None:
            interval = max(50, int(interval_steps))
            from_step = max(int(0.05 * max_steps), 200)
            until_step = int(until_frac * max_steps)
            from_cycle = max(1, (from_step + interval - 1) // interval)
            until_cycle = until_step // interval
            reset_every = max(2, -(-DECAY_MIN_STEPS // interval))
            return cls(
                interval_steps=interval,
                from_cycle=from_cycle,
                until_cycle=until_cycle,
                reset_every=reset_every,
                densify_every=1,
            )
        # LiteGS: vendor epoch cadence. Cycle index == epoch.
        total_epochs = max(1, int(math.ceil(max_steps / n_train)))
        long_run = total_epochs >= SHORT_RUN_EPOCHS
        densify_every = VENDOR_DENSIFY_EVERY if long_run else 1
        from_cycle = VENDOR_DENSIFY_FROM if long_run else 1
        until_cycle = max(from_cycle + 1, int(until_frac * total_epochs))
        reset_every = VENDOR_OPACITY_RESET if long_run else max(2, -(-DECAY_MIN_STEPS // n_train))
        return cls(
            interval_steps=n_train,
            from_cycle=from_cycle,
            until_cycle=until_cycle,
            reset_every=reset_every,
            densify_every=densify_every,
        )

    @property
    def active(self) -> bool:
        return self.until_cycle > self.from_cycle + 1

    def cycle(self, step: int) -> int:
        return int(step) // self.interval_steps

    def cycle_ends(self, step_after: int) -> bool:
        """True when ``step_after`` (steps completed so far) closes a cycle."""
        return step_after % self.interval_steps == 0

    def describe(self) -> str:
        if not self.active:
            return (
                f"densify off (run too short: {self.interval_steps}-step cycles, "
                f"window {self.from_cycle}..{self.until_cycle})"
            )
        n = self.until_cycle - self.from_cycle
        every = f" every {self.densify_every}" if self.densify_every > 1 else ""
        return (
            f"{n} cycles of {self.interval_steps} steps{every} "
            f"(epochs {self.from_cycle}..{self.until_cycle}), "
            f"opacity decay every {self.reset_every} cycles"
        )

    def apply(self, dp) -> None:
        """Set LiteGS DensifyParams so the cycle index plays the role of the epoch."""
        dp.densification_interval = int(self.densify_every)
        dp.opacity_reset_interval = int(self.reset_every)
        dp.densify_from = int(self.from_cycle)
        dp.densify_until = int(self.until_cycle) if self.active else int(self.from_cycle)


def sh_interval(max_steps: int, n_train: int = 0) -> int:
    """SH warm-up. Vendor LiteGS: one band every 5 epochs (``int(epoch/5)``)."""
    n_train = max(int(n_train), 0)
    if n_train > 0:
        return max(1, VENDOR_DENSIFY_EVERY * n_train)
    return int(max(200, min(1000, int(max_steps) // 10)))


def make_controller(kind: str, backend, *, budget: int, schedule: DensifySchedule, log=None):
    """``kind`` is a defaults.STRATEGIES label. ``backend`` is a LiteGSBackend."""
    if kind == D.STRATEGY_MCMC_COMPACT:
        from engine.splat.controllers.mcmc_compact import build as build_mcmc

        return build_mcmc(backend, budget=budget, schedule=schedule, log=log)
    from engine.splat.controllers.litegs_default import build as build_lite

    return build_lite(backend, budget=budget, schedule=schedule, log=log)
