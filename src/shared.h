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
    uvec4 numWorkGroups;
    uvec4 workGroupSize;
    uvec4 workGroupID;
    uvec4 localInvocationID;
    uvec4 globalInvocationID;
    uvec4 localInvocationIndex;
    uvec4 subgroup;
};

void save_data(Pixel *data, int width, int height, int depth);

#ifdef __cplusplus
}
#endif

#endif
