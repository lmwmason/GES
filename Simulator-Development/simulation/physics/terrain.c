#include "terrain.h"
#include <math.h>
#include <stddef.h>

#define TERRAIN_BUMP   0
#define TERRAIN_DIP    1
#define TERRAIN_RIPPLE 2

typedef struct {
    int type;
    double x0, length;   /* longitudinal extent [x0, x0+length], m */
    double amplitude;    /* peak height, m (always positive; DIP negates it) */
    double y0, y_sigma;  /* cross-track center offset and Gaussian falloff scale, m */
    double wavelength, phase; /* RIPPLE only */
} terrain_feature_t;

static terrain_feature_t g_features[TERRAIN_MAX_FEATURES];
static int g_feature_count = 0;

/* xorshift32 -- deterministic and trivial to reproduce bit-for-bit in
 * Python (see viz/terrain_ref.py), so the viewer can regenerate the exact
 * same terrain for a given seed without exchanging any file. Draw order
 * below must match terrain_ref.py's generate_features() exactly. */
static unsigned int g_prng_state = 1;

static unsigned int xorshift32(void) {
    unsigned int x = g_prng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_prng_state = x;
    return x;
}

static double rand01(void) {
    return xorshift32() / 4294967296.0; /* [0,1) */
}

static double rand_range(double lo, double hi) {
    return lo + (hi - lo) * rand01();
}

void terrain_init(unsigned int seed) {
    g_prng_state = seed ? seed : 0x9e3779b9u;
    for (int i = 0; i < 4; i++) xorshift32(); /* warm up */

    /* Deliberately huge: hundreds to a thousand+ features strung end to
     * end so a single episode's track spans kilometers of irregular
     * ground -- see find_feature_index() below for why terrain_height()
     * stays cheap even at this size. */
    g_feature_count = 800 + (int)(rand01() * 700.0); /* 800..1499 */
    if (g_feature_count > TERRAIN_MAX_FEATURES) g_feature_count = TERRAIN_MAX_FEATURES;

    double x_cursor = 4.0;
    for (int i = 0; i < g_feature_count; i++) {
        terrain_feature_t f;
        double type_roll = rand01();
        f.type = (type_roll < 0.4) ? TERRAIN_BUMP : (type_roll < 0.8) ? TERRAIN_DIP : TERRAIN_RIPPLE;

        f.length = rand_range(3.0, 8.0);
        f.x0 = x_cursor + rand_range(0.5, 3.0);
        x_cursor = f.x0 + f.length;

        f.amplitude = rand_range(0.05, 0.22);
        f.y0 = rand_range(-1.4, 1.4);
        f.y_sigma = rand_range(0.9, 2.2);
        f.wavelength = rand_range(1.2, 3.5);
        f.phase = rand_range(0.0, 2.0 * M_PI);

        g_features[i] = f;
    }
}

static double feature_height(const terrain_feature_t *f, double x, double y) {
    if (x < f->x0 || x > f->x0 + f->length) return 0.0;
    double u = (x - f->x0) / f->length;
    double envelope = 0.5 * (1.0 - cos(2.0 * M_PI * u)); /* 0 at edges, 1 at center */
    double dy = y - f->y0;
    double radial = exp(-(dy * dy) / (2.0 * f->y_sigma * f->y_sigma));

    double h;
    switch (f->type) {
        case TERRAIN_BUMP: h = f->amplitude * envelope; break;
        case TERRAIN_DIP:  h = -f->amplitude * envelope; break;
        case TERRAIN_RIPPLE:
        default:
            h = f->amplitude * envelope * sin(2.0 * M_PI * (x - f->x0) / f->wavelength + f->phase);
            break;
    }

    /* Secondary higher-frequency wobble layered on every feature so the
     * surface reads as rough/irregular instead of a few clean smooth
     * lobes. Heuristic, not tied to any real terrain dataset. */
    double roughness = 1.0 + 0.35 * sin(5.3 * (x - f->x0) + f->phase * 1.7) * envelope;
    return h * radial * roughness;
}

/* Features are generated with strictly increasing, non-overlapping
 * [x0, x0+length] windows (x_cursor only ever advances in terrain_init),
 * so at most one feature can be active at any x. That turns the lookup
 * into a binary search instead of a scan over all ~1000+ features --
 * without this, a long track would make every terrain_height() call (and
 * there are dozens per physics step, per rotor, per RK4 stage) scale with
 * the total feature count instead of O(log n). */
static int find_feature_index(double x) {
    int lo = 0, hi = g_feature_count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const terrain_feature_t *f = &g_features[mid];
        if (x < f->x0) hi = mid - 1;
        else if (x > f->x0 + f->length) lo = mid + 1;
        else return mid;
    }
    return -1;
}

double terrain_height(double x, double y) {
    int idx = find_feature_index(x);
    if (idx < 0) return 0.0;
    return feature_height(&g_features[idx], x, y);
}

void terrain_gradient(double x, double y, double *dhdx, double *dhdy) {
    const double eps = 0.02;
    *dhdx = (terrain_height(x + eps, y) - terrain_height(x - eps, y)) / (2.0 * eps);
    *dhdy = (terrain_height(x, y + eps) - terrain_height(x, y - eps)) / (2.0 * eps);
}

double terrain_curvature(double x, double y) {
    const double eps = 0.05;
    double h0 = terrain_height(x, y);
    double hxp = terrain_height(x + eps, y);
    double hxm = terrain_height(x - eps, y);
    double hyp = terrain_height(x, y + eps);
    double hym = terrain_height(x, y - eps);
    double d2hdx2 = (hxp - 2.0 * h0 + hxm) / (eps * eps);
    double d2hdy2 = (hyp - 2.0 * h0 + hym) / (eps * eps);
    return d2hdx2 + d2hdy2;
}
