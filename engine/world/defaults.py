"""World mesh numbers. Tuned on alpine_01; change only with a measured reason."""

from __future__ import annotations

GUIDE_RENDERER = "mesh"  # "mesh" or "points"; "points" restores the old dot guide
GUIDE_SUPERSAMPLE = 2
HIRES_SUPERSAMPLE = 2

# A triangle whose normal is almost perpendicular to the ray from its camera is a
# stretched skin across an occlusion edge. |cos| below this drops it.
GRAZE_COS = 0.02
DEPTH_RATIO = 1.6

GROUND_MIN_BELOW_DEG = 10.0
GROUND_RANSAC_ITERS = 400
GROUND_TOL_REL = 0.01
GROUND_SNAP_REL = 0.03
GROUND_MAX_TILT_DEG = 15.0
GROUND_MIN_INLIER_FRAC = 0.30
GROUND_SEED = 0
GROUND_NORMAL_DEG = 15.0
GROUND_FILL_REL = 0.05

PLANE_RANSAC_ITERS = 300
PLANE_TOL_REL = 0.01
PLANE_NORMAL_DEG = 10.0
PLANE_MIN_POINTS = 20000
PLANE_MAX = 6
PLANE_MIN_COMPONENT = 4000
EDGE_LOG_JUMP = 0.03

MESH_CANVAS = (2048, 1024)

LIFT_MIN_PIXELS = 256
FEATHER_PX = 8

BIN_SIZE = 128
# Alpine's plaza packs more than 200_000 faces into one 128px bin. Joint frame
# 19 needs 2_000_000. Below that the rasterizer leaves a square hole.
MAX_FACES_PER_BIN = 2_000_000
MAX_FACES_PER_BIN_CAP = 8_000_000
# Empty pixels whose ray meets the ground plane in this range take the panorama
# colour of that point. On alpine frame 10 the black pavement tears are those
# hits, and the tree holes are not.
GROUND_FILL_MIN = 0.05
GROUND_FILL_MAX = 40.0

# A full HiRes frame at the overflow cap is about 12 GB of bin lists. Tile past this.
RASTER_BIN_BUDGET = 3 * 2**30
HOLE_CLOSE_PX = 2
SLIVER_PX = 3
HOLE_GROW_PX = 4
GUIDE_FEATHER_PX = 8

GATE_PANO = 255
GATE_LAYER = 128
GATE_OWN = 0
