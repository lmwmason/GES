#ifndef MATH3D_H
#define MATH3D_H

typedef struct { double x, y, z; } vec3_t;

/* Hamilton convention, world_from_body: w,x,y,z */
typedef struct { double w, x, y, z; } quat_t;

vec3_t v3_add(vec3_t a, vec3_t b);
vec3_t v3_sub(vec3_t a, vec3_t b);
vec3_t v3_scale(vec3_t a, double s);
vec3_t v3_cross(vec3_t a, vec3_t b);
double v3_dot(vec3_t a, vec3_t b);
double v3_norm(vec3_t a);

quat_t q_identity(void);
quat_t q_normalize(quat_t q);
quat_t q_mul(quat_t a, quat_t b);
/* derivative of q given body-frame angular velocity omega */
quat_t q_derivative(quat_t q, vec3_t omega);
vec3_t q_rotate(quat_t q, vec3_t v);      /* rotate body-frame vector into world frame */
vec3_t q_rotate_inv(quat_t q, vec3_t v);  /* rotate world-frame vector into body frame */
void   q_to_euler(quat_t q, double *roll, double *pitch, double *yaw);

double clampd(double v, double lo, double hi);
double smoothstep01(double t); /* t clamped to [0,1], 3t^2-2t^3 */

#endif
