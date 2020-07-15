#ifndef COMPUTE_SHARED
#define COMPUTE_SHARED

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct uvec4 {
    uint32_t x, y, z, w;
};
struct Pixel {
    float r, g, b, a;
    struct uvec4 numWorkGroups;
    struct uvec4 workGroupSize;
    struct uvec4 workGroupID;
    struct uvec4 localInvocationID;
    struct uvec4 globalInvocationID;
    struct uvec4 localInvocationIndex;
    struct uvec4 subgroup;
};

void save_data(struct Pixel *data, int width, int height, int depth);

#ifdef __cplusplus
}
#endif

#endif
