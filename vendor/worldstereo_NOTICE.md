WorldStereo inference code is vendored at `vendor/worldstereo/` from
https://github.com/FuchengSu/WorldStereo commit `5ba8dc3d37c48ecef99bbd603815ced513a4a431`
(`update arxiv link`). The repository `LICENSE.md` is Apache-2.0.

`src/sp_utils/parallel_states.py` and `src/sp_utils/communications.py` also carry a
Tencent Hunyuan Community License header (HunyuanVideo-1.5). Those files are part of
the pinned WorldStereo tree. This checkout does not include
`Tencent-Hunyuan/HY-World-2.0`.

The weights are not in this tree. `hanshanxue/WorldStereo` has no licence file.
Which licence governs the weights is still UNKNOWN. Yik accepted that risk in
writing on 2026-10-29 (Index certainty log #59, PRD gate G2). The FP8 card’s
MIT sentence is not a settled licence.
