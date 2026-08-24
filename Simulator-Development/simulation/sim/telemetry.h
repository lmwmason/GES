#ifndef TELEMETRY_H
#define TELEMETRY_H

#include <stdint.h>

#pragma pack(push, 1)
typedef struct {
    double t;
    float pos[3];
    float quat[4];             /* w,x,y,z */
    float vel[3];
    float ang_vel[3];
    float motor_rpm[4];
    float motor_thrust[4];
    float ge_factor_truth[4];  /* physics ground-truth GE factor per rotor */
    float rangefinder_alt;
    float ge_ff_analytical;    /* firmware's flat-ground estimate */
    float ge_ff_ai;            /* firmware's AI-hook prediction */
    int32_t ge_ff_source;      /* 0=analytical,1=ai,2=disabled */
    float thrust_cmd_norm;
} telemetry_packet_t;
#pragma pack(pop)

typedef struct {
    int sock_fd;
} telemetry_t;

int  telemetry_init(telemetry_t *tm, const char *host, int port);
void telemetry_send(telemetry_t *tm, const telemetry_packet_t *pkt);
void telemetry_close(telemetry_t *tm);

#endif
