#ifndef RIGID_BODY_H
#define RIGID_BODY_H

#include "math3d.h"

typedef struct {
    vec3_t pos;    /* world frame [m], z-up */
    vec3_t vel;    /* world frame [m/s] */
    quat_t quat;   /* body-to-world rotation */
    vec3_t omega;  /* body-frame angular velocity [rad/s] */
} rigid_body_state_t;

/* User-supplied callback: given the current state, return net external
 * force (world frame, [N]) and net external torque about the CG
 * (body frame, [N*m]). Called multiple times per step (RK4 stages), so it
 * must be a pure function of *state (and whatever it captures via ctx). */
typedef void (*dynamics_forces_fn)(const rigid_body_state_t *state, void *ctx,
                                    vec3_t *force_world, vec3_t *torque_body);

void rigid_body_init(rigid_body_state_t *s, vec3_t pos0);

/* Classical RK4 integration step of the 6DOF rigid body equations,
 * including the gyroscopic term I*omega x omega. */
void rigid_body_step(rigid_body_state_t *s, double dt,
                      dynamics_forces_fn forces_fn, void *ctx,
                      double mass, vec3_t inertia_diag);

#endif
