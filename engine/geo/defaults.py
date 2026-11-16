"""Plot Camera defaults copied from SMPL CAMERA PLOT nodes.

SplatKit INPUT_TYPES at vendor pin f59de25: DatasetProject optional bool is ``reset``.
moge_level 9 is the graph widget, not fps. Overlay point_budget 4000 (min 500, max 40000).
P3: four SMPL knot sets (nodes 27, 122, 133, 144) + optional 5th (node 157).
"""

from __future__ import annotations

LOOK_MODES = ("look_forward", "look_at_target", "per_point_look")
LOOK_DEFAULT = "look_forward"

LENGTH_DEFAULT = 81
LENGTH_MIN = 9
LENGTH_MAX = 257
LENGTH_STEP = 4

MOGE_LEVEL = 9  # Plot Camera graph value. Do not treat as fps. HiRes stays 6 in P3.

POINT_BUDGET = 4000
POINT_BUDGET_MIN = 500
POINT_BUDGET_MAX = 40000

LOOK_AT_DEFAULT = "-0.108, 0.073, 1.953"

ANCHORS_DEFAULT = """0.000, 0.000, 0.000
0.199, 0.219, 0.488
-0.184, -0.093, 0.920
-0.065, 0.039, 1.404"""

DEFAULT_RAIL_COUNT = 4  # first-open straight rails + Suggest paths fill
STRAIGHT_RAIL_LENGTH = 1.5  # metres, editor frame, before the cloud shortens it
STRAIGHT_WALL_FRACTION = 0.8  # keep the default end inside the MoGe cloud
# Unit directions in the editor frame (+Y up, +Z = pano centre).
STRAIGHT_RAILS = (
    {"label": "forward", "direction": (0.0, 0.0, 1.0)},
    {"label": "back", "direction": (0.0, 0.0, -1.0)},
    {"label": "left", "direction": (-1.0, 0.0, 0.0)},
    {"label": "right", "direction": (1.0, 0.0, 0.0)},
)
MIN_RAILS = 1  # rail 1 (SMPL node 27) stays; user may remove 2–5
MAX_RAILS = 5  # graph CAMERA 1–4 + optional Add-to-dataset id 157
SUGGEST_COUNT = 4  # SplatKit suggest_paths default; function cap is 8

# SMPL demo rails — copy verbatim from docs/Votion_3DGS_2.0_Phase3.md (then let the user edit).
SMPL_RAILS = (
    {
        "node": 27,
        "orientation": "look_at_target",
        "look_at": LOOK_AT_DEFAULT,
        "label": "pond",
        "length": LENGTH_DEFAULT,
        "anchors": ANCHORS_DEFAULT,
    },
    {
        "node": 122,
        "orientation": "per_point_look",
        "look_at": LOOK_AT_DEFAULT,
        "label": "",
        "length": LENGTH_DEFAULT,
        "anchors": """0.000, 0.000, 0.000, 0.386, 0.021, 1.085
0.430, 0.021, 0.843, 1.784, 0.043, 1.931
0.819, 0.043, -0.292, 2.525, 0.032, -0.347
0.808, 0.032, -1.413, 2.411, 0.021, -1.983""",
    },
    {
        "node": 133,
        "orientation": "per_point_look",
        "look_at": LOOK_AT_DEFAULT,
        "label": "",
        "length": LENGTH_DEFAULT,
        "anchors": """0.000, 0.000, 0.000, 1.160, 0.202, -0.953
0.714, 0.423, -1.231, 0.918, 0.416, -1.870
0.223, 0.329, -1.412, 0.223, 0.215, -2.093
-0.408, 0.215, -1.474, -0.510, 0.098, -1.994""",
    },
    {
        "node": 144,
        "orientation": "per_point_look",
        "look_at": LOOK_AT_DEFAULT,
        "label": "",
        "length": LENGTH_DEFAULT,
        "anchors": """0.000, 0.000, 0.000, -1.195, 0.000, 0.124
-0.464, 0.387, 0.129, -1.564, 0.000, -0.245
-0.686, 0.151, -0.519, -1.483, 0.000, -0.740
-0.290, 0.199, -0.983, -0.893, 0.000, -1.584""",
    },
    {
        "node": 157,
        "orientation": "per_point_look",
        "look_at": LOOK_AT_DEFAULT,
        "label": "",
        "length": LENGTH_DEFAULT,
        "anchors": """0.000, 0.000, 0.000, -0.222, 0.000, -0.812
-0.395, 0.189, 0.356, -1.301, 0.124, 0.932
0.028, 0.539, 0.668, 0.390, 0.552, 1.611
0.225, 0.811, 0.083, 0.692, 0.772, -0.493""",
    },
)

HELP_GEOMETRY = """GEOMETRY
Compute geometry first. FLOOR / SIDE / Preview stay empty until then.
First open seeds four straight look_forward rails: forward, back, left, right.
Each is shortened to 80% of the free distance to the MoGe cloud so it does not start in a wall.
A clips-wall hint warns when an edited rail still goes past the cloud. It does not block Confirm.
Suggest paths stays as the fit-this-room alternative. Remove down to rail 1; Add fills back up to 5.
Active rail solid; others dimmed. Extra rails buy volume, not sharper views, and do not fix glass.
Star is origin, not draggable. look_forward + drag arrow switches that rail to per_point_look.
Play the mesh flight on the Preview pane before Confirm rails. Confirm gates Generate.
moge_level 9 is shared (not fps).
If the depth model changes, re-plot every rail. Generate stays blocked until those flights match the current depth.
"""
