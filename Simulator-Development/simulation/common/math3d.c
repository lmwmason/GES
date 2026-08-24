#include "math3d.h"
#include <math.h>

vec3_t v3_add(vec3_t a, vec3_t b) { return (vec3_t){a.x + b.x, a.y + b.y, a.z + b.z}; }
vec3_t v3_sub(vec3_t a, vec3_t b) { return (vec3_t){a.x - b.x, a.y - b.y, a.z - b.z}; }
vec3_t v3_scale(vec3_t a, double s) { return (vec3_t){a.x * s, a.y * s, a.z * s}; }
vec3_t v3_cross(vec3_t a, vec3_t b) {
    return (vec3_t){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
double v3_dot(vec3_t a, vec3_t b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double v3_norm(vec3_t a) { return sqrt(v3_dot(a, a)); }

quat_t q_identity(void) { return (quat_t){1.0, 0.0, 0.0, 0.0}; }

quat_t q_normalize(quat_t q) {
    double n = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n < 1e-12) return q_identity();
    double inv = 1.0 / n;
    return (quat_t){q.w * inv, q.x * inv, q.y * inv, q.z * inv};
}

quat_t q_mul(quat_t a, quat_t b) {
    return (quat_t){
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    };
}

quat_t q_derivative(quat_t q, vec3_t omega) {
    quat_t omega_q = {0.0, omega.x, omega.y, omega.z};
    quat_t dq = q_mul(q, omega_q);
    return (quat_t){0.5 * dq.w, 0.5 * dq.x, 0.5 * dq.y, 0.5 * dq.z};
}

vec3_t q_rotate(quat_t q, vec3_t v) {
    /* v' = q * v * q^-1, expanded closed-form */
    vec3_t u = {q.x, q.y, q.z};
    double s = q.w;
    vec3_t t = v3_scale(v3_cross(u, v), 2.0);
    vec3_t a = v3_add(v, v3_scale(t, s));
    vec3_t b = v3_cross(u, t);
    return v3_add(a, b);
}

vec3_t q_rotate_inv(quat_t q, vec3_t v) {
    quat_t qc = {q.w, -q.x, -q.y, -q.z};
    return q_rotate(qc, v);
}

void q_to_euler(quat_t q, double *roll, double *pitch, double *yaw) {
    double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    *roll = atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    if (fabs(sinp) >= 1.0) *pitch = copysign(M_PI / 2.0, sinp);
    else *pitch = asin(sinp);

    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    *yaw = atan2(siny_cosp, cosy_cosp);
}

double clampd(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

double smoothstep01(double t) {
    t = clampd(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
