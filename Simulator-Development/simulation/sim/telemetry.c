#include "telemetry.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int telemetry_init(telemetry_t *tm, const char *host, int port) {
    tm->sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (tm->sock_fd < 0) {
        perror("socket");
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "telemetry_init: bad host '%s'\n", host);
        return -1;
    }

    if (connect(tm->sock_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        return -1;
    }
    return 0;
}

void telemetry_send(telemetry_t *tm, const telemetry_packet_t *pkt) {
    ssize_t n = send(tm->sock_fd, pkt, sizeof(*pkt), 0);
    (void)n; /* UDP: drops are fine, the viewer only cares about the latest frame */
}

void telemetry_close(telemetry_t *tm) {
    if (tm->sock_fd >= 0) close(tm->sock_fd);
}
