#!/usr/bin/env bash
set -Eeuo pipefail

# Search the strongest VGGT-pointmap geometry-aware patch under sticker-like
# physical constraints. This keeps the same feature-L1 objective, but constrains
# placement to small, visible, planar surfaces instead of maximizing raw area.

export PLANE_MODE="${PLANE_MODE:-vggt_pointmap_surface}"
export TUM_GEOM_RUN_NAME="${TUM_GEOM_RUN_NAME:-tum10_vggt_pointmap_natural_geometry_feature_l3}"
export TUM_GEOM_MODEL="${TUM_GEOM_MODEL:-vggt_tum10_vggt_pointmap_natural_geometry_feature_l3}"

# A 25 cm sticker is much closer to a physical patch than the previous 60 cm
# debug plane. Size scales still allow the search to test nearby sizes.
export PLANE_WIDTH="${PLANE_WIDTH:-0.25}"
export PLANE_HEIGHT="${PLANE_HEIGHT:-0.25}"
export GEOMETRY_SIZE_SCALES="${GEOMETRY_SIZE_SCALES:-0.75,1.0,1.25}"
export GEOMETRY_ROLL_DEGREES="${GEOMETRY_ROLL_DEGREES:--30,-15,0,15,30}"

# Natural mode: keep only candidates that are visible enough, but do not cover
# an unnaturally large image area. Coverage is fraction of the VGGT input image.
export SURFACE_SCORE_MODE="${SURFACE_SCORE_MODE:-natural}"
export SURFACE_COVERAGE_MIN="${SURFACE_COVERAGE_MIN:-0.003}"
export SURFACE_COVERAGE_MAX="${SURFACE_COVERAGE_MAX:-0.06}"
export SURFACE_MIN_VISIBLE_FRAMES="${SURFACE_MIN_VISIBLE_FRAMES:-4}"
export SURFACE_MIN_VISIBILITY_RATIO="${SURFACE_MIN_VISIBILITY_RATIO:-0.5}"

# Search more candidate surfaces because the natural filter rejects many planes.
export FUSED_SURFACE_CANDIDATES="${FUSED_SURFACE_CANDIDATES:-256}"

exec bash "$(dirname "$0")/run_geometry_aware_tum10.sh"
