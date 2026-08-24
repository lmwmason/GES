#include "rigid_body.h"

typedef struct {
    vec3_t vel;
    vec3_t accel;
    quat_t dquat;
    vec3_t domega;
} rb_deriv_t;

void rigid_body_init(rigid_body_state_t *s, vec3_t pos0) {
    s->pos = pos0;
    s->vel = (vec3_t){0, 0, 0};
    s->quat = q_identity();
    s->omega = (vec3_t){0, 0, 0};
}

static rb_deriv_t compute_deriv(const rigid_body_state_t *s, double mass, vec3_t inertia_diag,
                                 dynamics_forces_fn forces_fn, void *ctx) {
    vec3_t force_world, torque_body;
    forces_fn(s, ctx, &force_world, &torque_body);

    rb_deriv_t d;
    d.vel = s->vel;
    d.accel = v3_scale(force_world, 1.0 / mass);
    d.dquat = q_derivative(s->quat, s->omega);

    vec3_t I_omega = { inertia_diag.x * s->omega.x,
                        inertia_diag.y * s->omega.y,
                        inertia_diag.z * s->omega.z };
    vec3_t gyroscopic = v3_cross(s->omega, I_omega);
    vec3_t net_torque = v3_sub(torque_body, gyroscopic);
    d.domega = (vec3_t){ net_torque.x / inertia_diag.x,
                          net_torque.y / inertia_diag.y,
                          net_torque.z / inertia_diag.z };
    return d;
}

static rigid_body_state_t state_add_scaled(const rigid_body_state_t *s, const rb_deriv_t *d, double h) {
    rigid_body_state_t out;
    out.pos = v3_add(s->pos, v3_scale(d->vel, h));
    out.vel = v3_add(s->vel, v3_scale(d->accel, h));
    out.quat.w = s->quat.w + d->dquat.w * h;
    out.quat.x = s->quat.x + d->dquat.x * h;
    out.quat.y = s->quat.y + d->dquat.y * h;
    out.quat.z = s->quat.z + d->dquat.z * h;
    out.omega = v3_add(s->omega, v3_scale(d->domega, h));
    return out;
}

void rigid_body_step(rigid_body_state_t *s, double dt,
                      dynamics_forces_fn forces_fn, void *ctx,
                      double mass, vec3_t inertia_diag) {
    rb_deriv_t k1 = compute_deriv(s, mass, inertia_diag, forces_fn, ctx);
    rigid_body_state_t s2 = state_add_scaled(s, &k1, dt * 0.5);
    rb_deriv_t k2 = compute_deriv(&s2, mass, inertia_diag, forces_fn, ctx);
    rigid_body_state_t s3 = state_add_scaled(s, &k2, dt * 0.5);
    rb_deriv_t k3 = compute_deriv(&s3, mass, inertia_diag, forces_fn, ctx);
    rigid_body_state_t s4 = state_add_scaled(s, &k3, dt);
    rb_deriv_t k4 = compute_deriv(&s4, mass, inertia_diag, forces_fn, ctx);

    double h6 = dt / 6.0;

    s->pos.x += h6 * (k1.vel.x + 2*k2.vel.x + 2*k3.vel.x + k4.vel.x);
    s->pos.y += h6 * (k1.vel.y + 2*k2.vel.y + 2*k3.vel.y + k4.vel.y);
    s->pos.z += h6 * (k1.vel.z + 2*k2.vel.z + 2*k3.vel.z + k4.vel.z);

    s->vel.x += h6 * (k1.accel.x + 2*k2.accel.x + 2*k3.accel.x + k4.accel.x);
    s->vel.y += h6 * (k1.accel.y + 2*k2.accel.y + 2*k3.accel.y + k4.accel.y);
    s->vel.z += h6 * (k1.accel.z + 2*k2.accel.z + 2*k3.accel.z + k4.accel.z);

    s->quat.w += h6 * (k1.dquat.w + 2*k2.dquat.w + 2*k3.dquat.w + k4.dquat.w);
    s->quat.x += h6 * (k1.dquat.x + 2*k2.dquat.x + 2*k3.dquat.x + k4.dquat.x);
    s->quat.y += h6 * (k1.dquat.y + 2*k2.dquat.y + 2*k3.dquat.y + k4.dquat.y);
    s->quat.z += h6 * (k1.dquat.z + 2*k2.dquat.z + 2*k3.dquat.z + k4.dquat.z);
    s->quat = q_normalize(s->quat);

    s->omega.x += h6 * (k1.domega.x + 2*k2.domega.x + 2*k3.domega.x + k4.domega.x);
    s->omega.y += h6 * (k1.domega.y + 2*k2.domega.y + 2*k3.domega.y + k4.domega.y);
    s->omega.z += h6 * (k1.domega.z + 2*k2.domega.z + 2*k3.domega.z + k4.domega.z);
}
