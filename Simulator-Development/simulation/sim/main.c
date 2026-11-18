#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>

#include "math3d.h"
#include "drone_params.h"
#include "sensor_types.h"
#include "fc_api.h"
#include "terrain.h"
#include "aero_ground_effect.h"
#include "motor_model.h"
#include "rigid_body.h"
#include "telemetry.h"

#define PHYSICS_HZ        1000.0
#define TELEMETRY_HZ      60.0
#define DEFAULT_DURATION  75.0
#define RANGEFINDER_MAX_M 3.0

typedef void (*fc_init_fn)(void);
typedef void (*fc_step_fn)(const sensor_packet_t *, actuator_packet_t *);
typedef void (*fc_get_debug_fn)(fc_debug_t *);
typedef void (*fc_set_diagnostic_ge_truth_fn)(double, double);
typedef void (*fc_set_diagnostic_prediction_bias_fn)(double);

typedef struct {
    const double *rpm; /* [4], held fixed across one physics step's RK4 stages */
} vehicle_ctx_t;

static double gauss_noise(double sigma) {
    if (sigma <= 0.0) return 0.0;
    double u1 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Shared by the RK4 dynamics callback and the sensor synthesizer: thrust
 * (with ground effect baked in), reaction torque, and body drag -- every
 * force on the vehicle except gravity, so it doubles as the accelerometer's
 * specific-force model (accel = (thrust+drag)/mass in body frame). */
static void compute_body_forces(const rigid_body_state_t *state, const double rpm[4],
                                 vec3_t *thrust_force_world, vec3_t *torque_body,
                                 vec3_t *drag_force_world, double ge_factor_out[4]) {
    vec3_t total_force_body = {0, 0, 0};
    vec3_t total_torque_body = {0, 0, 0};

    for (int i = 0; i < 4; i++) {
        vec3_t rotor_offset_body = { DP_MOTOR_LAYOUT[i].x, DP_MOTOR_LAYOUT[i].y, 0.0 };
        vec3_t rotor_world_pos = v3_add(state->pos, q_rotate(state->quat, rotor_offset_body));

        ge_result_t ge = ge_compute(rotor_world_pos, DP_ROTOR_RADIUS_M);
        if (ge_factor_out) ge_factor_out[i] = ge.factor;

        double thrust = motor_thrust_N(rpm[i]) * ge.factor;
        total_force_body.z += thrust;

        vec3_t r = rotor_offset_body;
        vec3_t F = {0, 0, thrust};
        total_torque_body = v3_add(total_torque_body, v3_cross(r, F));

        double q_reaction = motor_torque_Nm(rpm[i]);
        total_torque_body.z += -DP_MOTOR_LAYOUT[i].dir * q_reaction;
    }

    *thrust_force_world = q_rotate(state->quat, total_force_body);
    *torque_body = total_torque_body;

    double speed = v3_norm(state->vel);
    if (speed > 1e-6) {
        double drag_mag = 0.5 * DP_AIR_DENSITY * DP_BODY_DRAG_COEFF * DP_BODY_REF_AREA_M2 * speed * speed;
        *drag_force_world = v3_scale(state->vel, -drag_mag / speed);
    } else {
        *drag_force_world = (vec3_t){0, 0, 0};
    }
}

static void vehicle_forces(const rigid_body_state_t *state, void *ctx_v,
                            vec3_t *force_world, vec3_t *torque_body) {
    vehicle_ctx_t *ctx = (vehicle_ctx_t *)ctx_v;
    vec3_t thrust_w, drag_w;
    compute_body_forces(state, ctx->rpm, &thrust_w, torque_body, &drag_w, NULL);
    vec3_t gravity = {0, 0, -DP_MASS_KG * DP_GRAVITY};
    *force_world = v3_add(v3_add(thrust_w, gravity), drag_w);
}

static sensor_packet_t build_sensors(const rigid_body_state_t *rb, const double rpm[4],
                                      double t, double dt) {
    sensor_packet_t s;
    memset(&s, 0, sizeof(s));
    s.t = t;
    s.dt = dt;

    s.gyro[0] = rb->omega.x + gauss_noise(0.003);
    s.gyro[1] = rb->omega.y + gauss_noise(0.003);
    s.gyro[2] = rb->omega.z + gauss_noise(0.003);

    /* Accelerometer model: thrust is always exactly along body +z by
     * construction, so a literal specific-force calculation (thrust+drag,
     * rotated world->body) would come back out as (0,0,T/m) for *any*
     * attitude -- rotating a body-fixed force out to world and back is a
     * no-op, so it can never reveal tilt. Real low-cost complementary
     * filters sidestep this the same way we do here: assume near-hover
     * (thrust ~= weight) and treat the accelerometer as sensing the
     * world "up" reference rotated into the body frame, which is what
     * actually makes roll/pitch observable from accel + gyro alone. */
    vec3_t gravity_ref_world = {0.0, 0.0, DP_GRAVITY};
    vec3_t accel_body = q_rotate_inv(rb->quat, gravity_ref_world);

    s.accel[0] = accel_body.x + gauss_noise(0.05);
    s.accel[1] = accel_body.y + gauss_noise(0.05);
    s.accel[2] = accel_body.z + gauss_noise(0.05);

    double ground_h = terrain_height(rb->pos.x, rb->pos.y);
    double agl = rb->pos.z - ground_h; /* vertical cast; ignores beam tilt with attitude */
    s.rangefinder_alt = (agl <= RANGEFINDER_MAX_M) ? (agl + gauss_noise(0.01)) : NAN;
    s.baro_alt = rb->pos.z + gauss_noise(0.05);

    double total_current = 0.0;
    for (int i = 0; i < 4; i++) {
        s.motor_rpm_meas[i] = rpm[i] + gauss_noise(20.0);
        s.motor_current[i] = motor_current_A(rpm[i]) + gauss_noise(0.1);
        total_current += s.motor_current[i];
    }
    s.battery_voltage = 12.6 - 0.04 * total_current + gauss_noise(0.02);

    return s;
}

static char *default_fc_path(void) {
    static char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 32);
    if (n <= 0) {
        strcpy(buf, "./libfc.so");
        return buf;
    }
    buf[n] = '\0';
    char *slash = strrchr(buf, '/');
    if (slash) *(slash + 1) = '\0';
    strcat(buf, "libfc.so");
    return buf;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--fc PATH] [--host HOST] [--port PORT] [--duration SEC]\n"
        "          [--seed N] [--fast] [--no-telemetry] [--csv PATH]\n"
        "          [--oracle-bias X] [--oracle-pulse-start SEC] [--oracle-pulse-duration SEC]\n"
        "          [--oracle-aggregation arithmetic|thrust-weighted]\n"
        "          [--oracle-horizon SEC]\n"
        "          [--prediction-bias X]\n"
        "          [--ge-slope-exponent X] [--ge-curvature-gain X]\n"
        "          [--ge-base-model cheeseman|hayden]\n"
        "  --fc PATH        path to the firmware shared object (default: auto-detected next to this binary)\n"
        "  --seed N         terrain seed (default 1); different seeds -> different irregular terrain\n"
        "  --fast           skip real-time pacing, run as fast as the CPU allows (for dataset generation)\n"
        "  --no-telemetry   don't open/send the UDP telemetry socket\n"
        "  --csv PATH       write one row per telemetry tick to a CSV dataset file\n"
        "  --oracle-bias X  diagnostic signed bias added to oracle GE factor\n"
        "  --oracle-pulse-start SEC  apply oracle bias only after this time\n"
        "  --oracle-pulse-duration SEC  pulse width; requires pulse start\n"
        "  --oracle-aggregation MODE  collective truth reduction (default arithmetic)\n"
        "  --oracle-horizon SEC  kinematic terrain-preview horizon (diagnostic only)\n"
        "  --ge-slope-exponent X  heuristic slope exponent (default 1.0)\n"
        "  --ge-curvature-gain X  heuristic curvature gain (default 0.6)\n",
        prog);
}

int main(int argc, char **argv) {
    const char *fc_path = NULL;
    const char *udp_host = "127.0.0.1";
    int udp_port = 5555;
    double duration_s = DEFAULT_DURATION;
    unsigned int seed = 1;
    int fast_mode = 0;
    int telemetry_enabled = 1;
    const char *csv_path = NULL;
    double oracle_bias = 0.0;
    double oracle_pulse_start = -1.0;
    double oracle_pulse_duration = 0.0;
    const char *oracle_aggregation = "arithmetic";
    double oracle_horizon = 0.0;
    double prediction_bias = 0.0;
    double ge_slope_exponent = 1.0;
    double ge_curvature_gain = 0.6;
    ge_base_model_t ge_base_model = GE_BASE_CHEESEMAN;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fc") && i + 1 < argc) fc_path = argv[++i];
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) udp_host = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) udp_port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc) duration_s = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--fast")) fast_mode = 1;
        else if (!strcmp(argv[i], "--no-telemetry")) telemetry_enabled = 0;
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--oracle-bias") && i + 1 < argc) oracle_bias = atof(argv[++i]);
        else if (!strcmp(argv[i], "--oracle-pulse-start") && i + 1 < argc) oracle_pulse_start = atof(argv[++i]);
        else if (!strcmp(argv[i], "--oracle-pulse-duration") && i + 1 < argc) oracle_pulse_duration = atof(argv[++i]);
        else if (!strcmp(argv[i], "--oracle-aggregation") && i + 1 < argc) oracle_aggregation = argv[++i];
        else if (!strcmp(argv[i], "--oracle-horizon") && i + 1 < argc) oracle_horizon = atof(argv[++i]);
        else if (!strcmp(argv[i], "--prediction-bias") && i + 1 < argc) prediction_bias = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ge-slope-exponent") && i + 1 < argc) ge_slope_exponent = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ge-curvature-gain") && i + 1 < argc) ge_curvature_gain = atof(argv[++i]);
        else if (!strcmp(argv[i], "--ge-base-model") && i + 1 < argc) {
            const char *model = argv[++i];
            if (!strcmp(model, "cheeseman")) ge_base_model = GE_BASE_CHEESEMAN;
            else if (!strcmp(model, "hayden")) ge_base_model = GE_BASE_HAYDEN;
            else { fprintf(stderr, "[sim] invalid GE base model: %s\n", model); return 1; }
        }
        else if (!strcmp(argv[i], "--help")) { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "[sim] unknown argument: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }
    if ((oracle_pulse_start >= 0.0 && oracle_pulse_duration <= 0.0) ||
        (oracle_pulse_start < 0.0 && oracle_pulse_duration > 0.0)) {
        fprintf(stderr, "[sim] oracle pulse start and positive duration must be supplied together\n");
        return 1;
    }
    if (strcmp(oracle_aggregation, "arithmetic") && strcmp(oracle_aggregation, "thrust-weighted")) {
        fprintf(stderr, "[sim] invalid oracle aggregation: %s\n", oracle_aggregation);
        return 1;
    }
    if (oracle_horizon < 0.0 || oracle_horizon > 0.5) {
        fprintf(stderr, "[sim] oracle horizon must be in [0, 0.5] s\n");
        return 1;
    }
    if (!fc_path) fc_path = default_fc_path();

    printf("[sim] loading firmware: %s\n", fc_path);
    void *handle = dlopen(fc_path, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "[sim] dlopen failed: %s\n", dlerror());
        return 1;
    }
    fc_init_fn fc_init_p = (fc_init_fn)dlsym(handle, "fc_init");
    fc_step_fn fc_step_p = (fc_step_fn)dlsym(handle, "fc_step");
    fc_get_debug_fn fc_get_debug_p = (fc_get_debug_fn)dlsym(handle, "fc_get_debug");
    fc_set_diagnostic_ge_truth_fn fc_set_oracle_p =
        (fc_set_diagnostic_ge_truth_fn)dlsym(handle, "fc_set_diagnostic_ge_truth");
    fc_set_diagnostic_prediction_bias_fn fc_set_prediction_bias_p =
        (fc_set_diagnostic_prediction_bias_fn)dlsym(handle, "fc_set_diagnostic_prediction_bias");
    if (!fc_init_p || !fc_step_p || !fc_get_debug_p || !fc_set_oracle_p || !fc_set_prediction_bias_p) {
        fprintf(stderr, "[sim] dlsym failed: %s\n", dlerror());
        return 1;
    }
    fc_init_p();

    terrain_init(seed);
    ge_configure(ge_slope_exponent, ge_curvature_gain, ge_base_model);
    printf("[sim] terrain seed: %u\n", seed);

    telemetry_t tm;
    if (telemetry_enabled) {
        if (telemetry_init(&tm, udp_host, udp_port) != 0) {
            fprintf(stderr, "[sim] telemetry_init failed\n");
            return 1;
        }
        printf("[sim] streaming telemetry to %s:%d\n", udp_host, udp_port);
    }

    FILE *csv_fp = NULL;
    if (csv_path) {
        csv_fp = fopen(csv_path, "w");
        if (!csv_fp) {
            fprintf(stderr, "[sim] failed to open csv file: %s\n", csv_path);
            return 1;
        }
        fprintf(csv_fp,
            "seed,t,x,y,z,"
            "gyro_x,gyro_y,gyro_z,"
            "rangefinder_alt,baro_alt,"
            "motor_rpm_0,motor_rpm_1,motor_rpm_2,motor_rpm_3,"
            "motor_current_0,motor_current_1,motor_current_2,motor_current_3,"
            "thrust_cmd_norm,ge_ff_analytical,"
            "ge_factor_truth_0,ge_factor_truth_1,ge_factor_truth_2,ge_factor_truth_3,"
            "ge_factor_truth_mean,ge_ff_ai,ge_ff_source,ge_ff_applied,"
            "ge_oracle_truth,ge_oracle_bias,ge_ff_clamped,true_agl,min_rotor_clearance\n");
        printf("[sim] writing dataset csv: %s\n", csv_path);
    }

    rigid_body_state_t rb;
    rigid_body_init(&rb, (vec3_t){0.0, 0.0, 0.08});

    motor_state_t motors[4];
    for (int i = 0; i < 4; i++) motor_init(&motors[i]);

    vec3_t inertia_diag = {DP_INERTIA_XX, DP_INERTIA_YY, DP_INERTIA_ZZ};

    const double dt = 1.0 / PHYSICS_HZ;
    const double telemetry_dt = 1.0 / TELEMETRY_HZ;
    double t = 0.0;
    double next_telemetry_t = 0.0;

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    while (t < duration_s) {
        double rpm_now[4];
        for (int i = 0; i < 4; i++) rpm_now[i] = motors[i].rpm;

        sensor_packet_t sp = build_sensors(&rb, rpm_now, t, dt);
        double oracle_truth[4];
        for (int i = 0; i < 4; i++) {
            vec3_t off = {DP_MOTOR_LAYOUT[i].x, DP_MOTOR_LAYOUT[i].y, 0.0};
            vec3_t preview_pos = v3_add(rb.pos, v3_scale(rb.vel, oracle_horizon));
            vec3_t rotor_pos = v3_add(preview_pos, q_rotate(rb.quat, off));
            ge_result_t ge = ge_compute(rotor_pos, DP_ROTOR_RADIUS_M);
            oracle_truth[i] = ge.factor;
        }
        double oracle_mean = 0.25 * (oracle_truth[0] + oracle_truth[1] + oracle_truth[2] + oracle_truth[3]);
        if (!strcmp(oracle_aggregation, "thrust-weighted")) {
            double numerator = 0.0, denominator = 0.0;
            for (int i = 0; i < 4; i++) {
                double free_air_thrust = motor_thrust_N(rpm_now[i]);
                numerator += free_air_thrust * oracle_truth[i];
                denominator += free_air_thrust;
            }
            if (denominator > 1e-12) oracle_mean = numerator / denominator;
        }
        double active_oracle_bias = oracle_bias;
        double active_prediction_bias = prediction_bias;
        if (oracle_pulse_start >= 0.0 &&
            !(t >= oracle_pulse_start && t < oracle_pulse_start + oracle_pulse_duration)) {
            active_oracle_bias = 0.0;
            active_prediction_bias = 0.0;
        }
        fc_set_oracle_p(oracle_mean, active_oracle_bias);
        fc_set_prediction_bias_p(active_prediction_bias);
        actuator_packet_t ap;
        fc_step_p(&sp, &ap);

        for (int i = 0; i < 4; i++) motor_update(&motors[i], ap.motor_cmd[i], dt);

        double rpm_step[4];
        for (int i = 0; i < 4; i++) rpm_step[i] = motors[i].rpm;
        vehicle_ctx_t ctx = { .rpm = rpm_step };
        rigid_body_step(&rb, dt, vehicle_forces, &ctx, DP_MASS_KG, inertia_diag);

        double ground_h = terrain_height(rb.pos.x, rb.pos.y);
        if (rb.pos.z < ground_h + 0.08) {
            rb.pos.z = ground_h + 0.08;
            if (rb.vel.z < 0.0) rb.vel.z = 0.0;
        }

        t += dt;

        if (t >= next_telemetry_t) {
            double ge_truth[4];
            compute_body_forces(&rb, rpm_step, &(vec3_t){0}, &(vec3_t){0}, &(vec3_t){0}, ge_truth);
            double min_rotor_clearance = INFINITY;
            for (int i = 0; i < 4; i++) {
                vec3_t off = {DP_MOTOR_LAYOUT[i].x, DP_MOTOR_LAYOUT[i].y, 0.0};
                vec3_t rotor_pos = v3_add(rb.pos, q_rotate(rb.quat, off));
                double clearance = ge_compute(rotor_pos, DP_ROTOR_RADIUS_M).clearance;
                if (clearance < min_rotor_clearance) min_rotor_clearance = clearance;
            }
            double true_agl = rb.pos.z - terrain_height(rb.pos.x, rb.pos.y);

            fc_debug_t dbg;
            fc_get_debug_p(&dbg);

            telemetry_packet_t pkt;
            memset(&pkt, 0, sizeof(pkt));
            pkt.t = t;
            pkt.pos[0] = (float)rb.pos.x; pkt.pos[1] = (float)rb.pos.y; pkt.pos[2] = (float)rb.pos.z;
            pkt.quat[0] = (float)rb.quat.w; pkt.quat[1] = (float)rb.quat.x;
            pkt.quat[2] = (float)rb.quat.y; pkt.quat[3] = (float)rb.quat.z;
            pkt.vel[0] = (float)rb.vel.x; pkt.vel[1] = (float)rb.vel.y; pkt.vel[2] = (float)rb.vel.z;
            pkt.ang_vel[0] = (float)rb.omega.x; pkt.ang_vel[1] = (float)rb.omega.y; pkt.ang_vel[2] = (float)rb.omega.z;
            for (int i = 0; i < 4; i++) {
                pkt.motor_rpm[i] = (float)rpm_step[i];
                pkt.motor_thrust[i] = (float)(motor_thrust_N(rpm_step[i]) * ge_truth[i]);
                pkt.ge_factor_truth[i] = (float)ge_truth[i];
            }
            pkt.rangefinder_alt = (float)sp.rangefinder_alt;
            pkt.ge_ff_analytical = (float)dbg.ge_ff_analytical;
            pkt.ge_ff_ai = (float)dbg.ge_ff_ai;
            pkt.ge_ff_source = dbg.ge_ff_source;
            pkt.thrust_cmd_norm = (float)dbg.thrust_cmd_norm;

            if (telemetry_enabled) telemetry_send(&tm, &pkt);

            if (csv_fp) {
                double ge_mean = 0.25 * (ge_truth[0] + ge_truth[1] + ge_truth[2] + ge_truth[3]);
                fprintf(csv_fp,
                    "%u,%.4f,%.4f,%.4f,%.4f,"
                    "%.5f,%.5f,%.5f,"
                    "%.4f,%.4f,"
                    "%.2f,%.2f,%.2f,%.2f,"
                    "%.4f,%.4f,%.4f,%.4f,"
                    "%.4f,%.4f,"
                    "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d,"
                    "%.5f,%.5f,%.5f,%d,%.5f,%.5f\n",
                    seed, t, rb.pos.x, rb.pos.y, rb.pos.z,
                    sp.gyro[0], sp.gyro[1], sp.gyro[2],
                    sp.rangefinder_alt, sp.baro_alt,
                    sp.motor_rpm_meas[0], sp.motor_rpm_meas[1], sp.motor_rpm_meas[2], sp.motor_rpm_meas[3],
                    sp.motor_current[0], sp.motor_current[1], sp.motor_current[2], sp.motor_current[3],
                    dbg.thrust_cmd_norm, dbg.ge_ff_analytical,
                    ge_truth[0], ge_truth[1], ge_truth[2], ge_truth[3], ge_mean,
                    dbg.ge_ff_ai, dbg.ge_ff_source,
                    dbg.ge_ff_applied, dbg.ge_oracle_truth, dbg.ge_oracle_bias,
                    dbg.ge_ff_clamped, true_agl, min_rotor_clearance);
            }

            next_telemetry_t += telemetry_dt;
        }

        if (!fast_mode) {
            struct timespec t_now;
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double elapsed = (t_now.tv_sec - t_start.tv_sec) + (t_now.tv_nsec - t_start.tv_nsec) / 1e9;
            double behind = t - elapsed;
            if (behind > 0.0) {
                struct timespec ts = { .tv_sec = (time_t)behind, .tv_nsec = (long)((behind - (time_t)behind) * 1e9) };
                nanosleep(&ts, NULL);
            }
        }
    }

    if (telemetry_enabled) telemetry_close(&tm);
    if (csv_fp) fclose(csv_fp);
    dlclose(handle);
    printf("[sim] done, t=%.1fs\n", t);
    return 0;
}
