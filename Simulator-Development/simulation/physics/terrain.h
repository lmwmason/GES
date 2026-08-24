#ifndef TERRAIN_H
#define TERRAIN_H

/* Ground-truth terrain height field. This lives entirely on the physics
 * side of the simulator -- the firmware (physics/../firmware) never
 * includes this header. It only ever perceives the ground indirectly
 * through the rangefinder/IMU/motor-current entries of sensor_packet_t.
 *
 * The terrain is procedurally generated from an integer seed: a run of
 * irregular bump/dip/ripple features with randomized position, size,
 * sign, cross-track offset and (for ripples) wavelength/phase, each also
 * carrying a secondary higher-frequency wobble so the surface reads as
 * rough rather than a few clean smooth lobes. Different seeds -> visibly
 * different terrain, which is what dataset generation (many episodes,
 * each flying over a different ground shape) is built around. */

#define TERRAIN_MAX_FEATURES 1500

/* Call once before any terrain_height/gradient/curvature call. seed==0 is
 * remapped to a fixed nonzero value internally (xorshift's fixed point). */
void terrain_init(unsigned int seed);

/* Elevation at world (x,y) [m]. z-up, flat baseline = 0. */
double terrain_height(double x, double y);

/* Local surface gradient (dh/dx, dh/dy) via central differences. */
void terrain_gradient(double x, double y, double *dhdx, double *dhdy);

/* Local mean curvature approximation (Laplacian d2h/dx2 + d2h/dy2).
 * Positive = concave/bowl (traps recirculating downwash),
 * negative = convex/dome (downwash spills off the sides). */
double terrain_curvature(double x, double y);

#endif
