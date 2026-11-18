#include "aero_ground_effect.h"
#include "terrain.h"
#include <math.h>

static double g_slope_exponent = 1.0;
static double g_curvature_gain = 0.6;
static ge_base_model_t g_base_model = GE_BASE_CHEESEMAN;

double ge_flat_factor(double z, double rotor_radius_m, ge_base_model_t base_model) {
    double z_min = rotor_radius_m * 0.50;
    if (z < z_min) z = z_min;
    if (base_model == GE_BASE_HAYDEN) {
        double z_over_d = fmax(z / (2.0 * rotor_radius_m), 0.15);
        double fg = 1.0 / (0.9926 + 0.03794 / (z_over_d * z_over_d));
        fg = fmin(fg, 1.0);
        return pow(fg, -2.0 / 3.0);
    }
    double ratio = rotor_radius_m / (4.0 * z);
    return 1.0 / (1.0 - ratio * ratio);
}

void ge_configure(double slope_exponent, double curvature_gain, ge_base_model_t base_model) {
    g_slope_exponent = slope_exponent > 0.0 ? slope_exponent : 1.0;
    g_curvature_gain = curvature_gain >= 0.0 ? curvature_gain : 0.6;
    g_base_model = base_model;
}

ge_result_t ge_compute(vec3_t rotor_world_pos, double rotor_radius_m) {
    ge_result_t r;

    double terr_h = terrain_height(rotor_world_pos.x, rotor_world_pos.y);
    double z = rotor_world_pos.z - terr_h;
    double z_min = rotor_radius_m * 0.50; /* avoid singularity right at the surface */
    if (z < z_min) z = z_min;
    r.clearance = z;

    double factor_flat;
    if (g_base_model == GE_BASE_HAYDEN) {
        /* Hayden empirical induced-power model as documented in NASA NDARC
         * TP-20220000355, converted to thrust multiplier kappa=f_g^(-2/3).
         * NDARC restricts the expression to z/D >= 0.15 and f_g <= 1. */
        factor_flat = ge_flat_factor(z, rotor_radius_m, GE_BASE_HAYDEN);
    } else {
        /* Cheeseman & Bennett (1955) flat, level-ground IGE thrust ratio. */
        factor_flat = ge_flat_factor(z, rotor_radius_m, GE_BASE_CHEESEMAN);
    }

    double dhdx, dhdy;
    terrain_gradient(rotor_world_pos.x, rotor_world_pos.y, &dhdx, &dhdy);
    double slope = atan(sqrt(dhdx * dhdx + dhdy * dhdy));
    r.slope_rad = slope;

    /* Sloped ground breaks up the recirculating vortex ring that produces
     * the IGE benefit, so scale the augmentation down with slope angle.
     * Heuristic extension -- not from a specific validated dataset;
     * an exact answer would require a CFD solve. */
    double augmentation = (factor_flat - 1.0) * pow(fmax(0.0, cos(slope)), g_slope_exponent);

    double kappa = terrain_curvature(rotor_world_pos.x, rotor_world_pos.y);
    r.curvature = kappa;

    /* Concave ground (bowl, kappa>0) traps downwash and boosts the effect;
     * convex ground (dome, kappa<0) lets it spill off the sides and
     * reduces it. Gain tuned so curvature radii ~ rotor radius give an
     * O(1) correction, decaying for gentler curves. */
    double factor_curv = 1.0 + g_curvature_gain * kappa * rotor_radius_m * rotor_radius_m;
    factor_curv = clampd(factor_curv, 0.5, 1.8);

    double factor = 1.0 + augmentation * factor_curv;
    r.factor = clampd(factor, 0.85, 1.6);
    return r;
}
