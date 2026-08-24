#include "aero_ground_effect.h"
#include "terrain.h"
#include <math.h>

ge_result_t ge_compute(vec3_t rotor_world_pos, double rotor_radius_m) {
    ge_result_t r;

    double terr_h = terrain_height(rotor_world_pos.x, rotor_world_pos.y);
    double z = rotor_world_pos.z - terr_h;
    double z_min = rotor_radius_m * 0.50; /* avoid singularity right at the surface */
    if (z < z_min) z = z_min;
    r.clearance = z;

    /* Cheeseman & Bennett (1955) flat, level-ground IGE thrust ratio. */
    double ratio = rotor_radius_m / (4.0 * z);
    double factor_flat = 1.0 / (1.0 - ratio * ratio);

    double dhdx, dhdy;
    terrain_gradient(rotor_world_pos.x, rotor_world_pos.y, &dhdx, &dhdy);
    double slope = atan(sqrt(dhdx * dhdx + dhdy * dhdy));
    r.slope_rad = slope;

    /* Sloped ground breaks up the recirculating vortex ring that produces
     * the IGE benefit, so scale the augmentation down with slope angle.
     * Heuristic extension -- not from a specific validated dataset;
     * an exact answer would require a CFD solve. */
    double augmentation = (factor_flat - 1.0) * cos(slope);

    double kappa = terrain_curvature(rotor_world_pos.x, rotor_world_pos.y);
    r.curvature = kappa;

    /* Concave ground (bowl, kappa>0) traps downwash and boosts the effect;
     * convex ground (dome, kappa<0) lets it spill off the sides and
     * reduces it. Gain tuned so curvature radii ~ rotor radius give an
     * O(1) correction, decaying for gentler curves. */
    double curv_gain = 0.6;
    double factor_curv = 1.0 + curv_gain * kappa * rotor_radius_m * rotor_radius_m;
    factor_curv = clampd(factor_curv, 0.5, 1.8);

    double factor = 1.0 + augmentation * factor_curv;
    r.factor = clampd(factor, 0.85, 1.6);
    return r;
}
