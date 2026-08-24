#ifndef DRONE_PARAMS_H
#define DRONE_PARAMS_H

/* Airframe constants. Shared by the physics engine (ground truth) and the
 * firmware (as if they were compiled-in vehicle parameters on a real FC).
 * Nothing terrain- or environment-related belongs in here. */

#define DP_MASS_KG          1.50
#define DP_ARM_LENGTH_M     0.18      /* motor offset from CG, X layout   */
#define DP_ROTOR_RADIUS_M   0.127     /* 10-inch prop diameter            */
#define DP_INERTIA_XX       0.0208
#define DP_INERTIA_YY       0.0208
#define DP_INERTIA_ZZ       0.0385

#define DP_GRAVITY          9.81
#define DP_AIR_DENSITY      1.225

/* motor/prop: thrust = KT * rpm^2 [N], drag torque = KQ * rpm^2 [N*m] */
#define DP_MOTOR_KT         1.472e-7
#define DP_MOTOR_KQ         2.940e-9
#define DP_MOTOR_RPM_MAX    9000.0
#define DP_MOTOR_TAU_S      0.030    /* first-order spin-up/down time constant */

/* body aerodynamic drag (simple quadratic drag on the airframe body) */
#define DP_BODY_DRAG_COEFF  0.30
#define DP_BODY_REF_AREA_M2 0.03

/* Motor layout, X configuration, order: FR, RL, FL, RR (Betaflight-style).
 * dir: +1 = CCW, -1 = CW (reaction torque sign) */
typedef struct {
    double x, y;   /* body-frame offsets [m], x-forward, y-right */
    int dir;
} motor_layout_t;

static const motor_layout_t DP_MOTOR_LAYOUT[4] = {
    { DP_ARM_LENGTH_M * 0.70710678, -DP_ARM_LENGTH_M * 0.70710678, -1 }, /* 0: front-right, CW  */
    { -DP_ARM_LENGTH_M * 0.70710678,  DP_ARM_LENGTH_M * 0.70710678, -1 }, /* 1: rear-left,   CW  */
    { DP_ARM_LENGTH_M * 0.70710678,  DP_ARM_LENGTH_M * 0.70710678,  1 }, /* 2: front-left,  CCW */
    { -DP_ARM_LENGTH_M * 0.70710678, -DP_ARM_LENGTH_M * 0.70710678,  1 }, /* 3: rear-right,  CCW */
};

#endif
