#ifndef AERO_GROUND_EFFECT_H
#define AERO_GROUND_EFFECT_H

#include "math3d.h"

/* Ground-truth (physics-side) in-ground-effect model for a single rotor.
 * Base term is the classic Cheeseman-Bennett flat-ground IGE ratio,
 * extended with heuristic slope/curvature corrections so thrust responds
 * to non-flat terrain (bumps/dips/ripples). This is a semi-empirical
 * engineering approximation, not a CFD solve -- see aero_ground_effect.c
 * for the reasoning behind each correction term. */
typedef struct {
    double clearance;   /* rotor height above local terrain [m] */
    double factor;      /* thrust multiplier applied on top of free-air thrust, >=1 typical */
    double slope_rad;   /* local terrain slope magnitude under the rotor [rad] */
    double curvature;   /* local terrain Laplacian under the rotor [1/m] */
} ge_result_t;

ge_result_t ge_compute(vec3_t rotor_world_pos, double rotor_radius_m);
void ge_configure(double slope_exponent, double curvature_gain);

#endif
