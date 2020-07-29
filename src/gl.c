#define GL_GLEXT_PROTOTYPES

#include <assert.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <fcntl.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <gbm.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "shared.h"

struct {
    bool enabled;
    bool show_csv;
    FILE *statsFile;
    struct {
        unsigned queryHandle;
        unsigned dataSize;
        unsigned off_threads;
        unsigned off_thread_occupancy_pct;
        unsigned off_time_ns;
    } compute_metrics_basic;

    struct {
        unsigned queryHandle;
        unsigned dataSize;
        unsigned off_cs_invocations;
    } pipeline_statistics;

    bool dbg;
} perf;

static void
query_query(char *name, bool pipeline)
{
    unsigned queryId;
    glGetPerfQueryIdByNameINTEL(name, &queryId);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr,
                "Query %s not found. Disable performance queries with PERF_ENABLED=0\n",
                name);
        assert(0);
    }
    if (perf.dbg)
        printf("queryId: %u\n", queryId);

    char queryName[4096];
    unsigned dataSize, noCounters, noInstances, capsMask;
    glGetPerfQueryInfoINTEL(queryId, sizeof(queryName), queryName, &dataSize,
                            &noCounters, &noInstances, &capsMask);
    assert(glGetError() == GL_NO_ERROR);

    if (perf.dbg)
        printf("query name: %s, data size: %u\n", queryName, dataSize);
    if (pipeline)
        perf.pipeline_statistics.dataSize = dataSize;
    else
        perf.compute_metrics_basic.dataSize = dataSize;

    for (int counterId = 1; counterId <= noCounters; counterId++) {
        uint counterOffset;
        uint counterDataSize;
        uint counterTypeEnum;
        uint counterDataTypeEnum;
        uint64_t rawCounterMaxValue;
        char counterName[32];
        char counterDesc[256];

        glGetPerfCounterInfoINTEL(
                queryId,
                counterId,
                sizeof(counterName),
                counterName,
                sizeof(counterDesc),
                counterDesc,
                &counterOffset,
                &counterDataSize,
                &counterTypeEnum,
                &counterDataTypeEnum,
                &rawCounterMaxValue);
        assert(glGetError() == GL_NO_ERROR);

        if (strcmp(counterName, "CS Threads Dispatched") == 0) {
            assert(!pipeline);
            perf.compute_metrics_basic.off_threads = counterOffset;
            assert(counterDataSize == 8);
            assert(counterDataTypeEnum == GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL);
            assert(counterTypeEnum == GL_PERFQUERY_COUNTER_EVENT_INTEL);
        } else if (strcmp(counterName, "EU Thread Occupancy") == 0) {
            assert(!pipeline);
            perf.compute_metrics_basic.off_thread_occupancy_pct = counterOffset;
            assert(counterDataSize == 4);
            assert(counterDataTypeEnum == GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL);
            assert(counterTypeEnum == GL_PERFQUERY_COUNTER_RAW_INTEL);
        } else if (strcmp(counterName, "GPU Time Elapsed") == 0) {
            assert(!pipeline);
            perf.compute_metrics_basic.off_time_ns = counterOffset;
            assert(counterDataSize == 8);
            assert(counterDataTypeEnum == GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL);
            assert(counterTypeEnum == GL_PERFQUERY_COUNTER_RAW_INTEL);
        } else if (strcmp(counterName, "N compute shader invocations") == 0) {
            assert(pipeline);
            perf.pipeline_statistics.off_cs_invocations = counterOffset;
            assert(counterDataSize == 8);
            assert(counterDataTypeEnum == GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL);
            assert(counterTypeEnum == GL_PERFQUERY_COUNTER_RAW_INTEL);
        }

        if (perf.dbg)
            printf("id: %2u, name: %32s, off: %3u, datasize: %u\n",
                    counterId, counterName, counterOffset, counterDataSize);
    }

    if (pipeline)
        glCreatePerfQueryINTEL(queryId, &perf.pipeline_statistics.queryHandle);
    else
        glCreatePerfQueryINTEL(queryId, &perf.compute_metrics_basic.queryHandle);
    assert(glGetError() == GL_NO_ERROR);
}

int
main(int argc, char *argv[])
{
    if (argc != 9) {
        fprintf(stderr, "not enough arguments\n");
        exit(2);
    }

    int WIDTH = atoi(argv[3]);
    int HEIGHT = atoi(argv[4]);
    int DEPTH = atoi(argv[5]);
    int WORKGROUP_SIZE_X = atoi(argv[6]);
    int WORKGROUP_SIZE_Y = atoi(argv[7]);
    int WORKGROUP_SIZE_Z = atoi(argv[8]);
    const char *tmp;

    tmp = getenv("PERF_ENABLED");
    perf.enabled = tmp == NULL || atoi(tmp) > 0;
    if (perf.enabled) {
        tmp = getenv("CSV");
        perf.show_csv = tmp != NULL && atoi(tmp) > 0;

        if (perf.show_csv) {
            perf.statsFile = fopen("stats.csv", "w");
            if (!perf.statsFile) {
                perror("fopen stats.csv");
                exit(2);
            }
            fprintf(perf.statsFile, "x:int,y:int,z:int,time_ns:int,threads:int,invocations:int,simd:int,thread_occupancy_pct:int,cpu_time_ns:int\n");
        }
    }

    if (WORKGROUP_SIZE_X == 0 || WORKGROUP_SIZE_Y == 0 || WORKGROUP_SIZE_Z == 0||
            WIDTH == 0 || HEIGHT == 0 || DEPTH == 0)
        abort();

    tmp = getenv("USE_VARIABLE_GROUP_SIZE");
    bool variable_group_size = tmp != NULL && atoi(tmp) > 0;

    int fd = open(argv[1], O_RDWR);
    if (fd < 0) {
        perror("open");
        exit(2);
    }

    struct gbm_device *gbm = gbm_create_device(fd);
    if (!gbm) {
        perror("gbm_create_device");
        exit(2);
    }

    EGLDisplay disp = eglGetPlatformDisplay(EGL_PLATFORM_GBM_MESA, gbm, NULL);
    if (!disp) {
        perror("eglGetPlatformDisplay");
        exit(2);
    }

    if (!eglInitialize(disp, NULL, NULL)) {
        perror("eglInitialize");
        exit(2);
    }

    const char *exts = eglQueryString(disp, EGL_EXTENSIONS);

    if (strstr(exts, "EGL_KHR_create_context") == NULL) {
        fprintf(stderr, "no support for EGL_KHR_create_context\n");
        exit(2);
    }
    if (strstr(exts, "EGL_KHR_surfaceless_context") == NULL) {
        fprintf(stderr, "no support for EGL_KHR_surfaceless_context\n");
        exit(2);
    }

    EGLint cfg_attrs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE };

    EGLConfig cfg;
    EGLint count;

    if (!eglChooseConfig(disp, cfg_attrs, &cfg, 1, &count)) {
        perror("eglChooseConfig");
        exit(2);
    }

    if (!eglBindAPI(EGL_OPENGL_API)) {
        perror("eglBindAPI");
        exit(2);
    }

    EGLint ctx_attrs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 5,
            EGL_NONE
    };

    EGLContext ctx = eglCreateContext(disp, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (ctx == EGL_NO_CONTEXT) {
        perror("eglCreateContext");
        exit(2);
    }

    if (!eglMakeCurrent(disp, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        perror("eglMakeCurrent");
        exit(2);
    }

    size_t bufferSize = sizeof(struct Pixel) * WIDTH * HEIGHT * DEPTH;
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    assert(glGetError() == GL_NO_ERROR);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    assert(glGetError() == GL_NO_ERROR);

    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, NULL, GL_STATIC_READ);
    assert(glGetError() == GL_NO_ERROR);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
    assert(glGetError() == GL_NO_ERROR);

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    if (shader == 0) {
        fprintf(stderr, "glCreateShader: 0x%x\n", glGetError());
        exit(2);
    }

    FILE *f = fopen(argv[2], "r");
    if (!f) {
        perror("fopen");
        exit(2);
    }
    struct stat st;
    if (fstat(fileno(f), &st)) {
        perror("fstat");
        exit(2);
    }

    char *shader_src = malloc(st.st_size + 1);
    if (!shader_src) {
        perror("malloc");
        exit(2);
    }

    size_t rem = st.st_size;
    size_t off = 0;
    while (rem > 0) {
        size_t r = fread(shader_src + off, 1, rem, f);
        if (r == 0) {
            fprintf(stderr, "fread: %zu %d %d\n", r, feof(f), ferror(f));
            exit(2);
        }

        off += r;
        rem -= r;
    }
    shader_src[st.st_size] = 0;

    char *pos;
    while ((pos = strstr(shader_src, "WIDTH")) != NULL) {
        sprintf(pos, "%-4d", WIDTH);
        *(pos + 4) = ' ';
    }
    while ((pos = strstr(shader_src, "HEIGHT")) != NULL) {
        sprintf(pos, "%-5d", HEIGHT);
        *(pos + 5) = ' ';
    }
    while ((pos = strstr(shader_src, "DEPTH")) != NULL) {
        sprintf(pos, "%-4d", DEPTH);
        *(pos + 4) = ' ';
    }
    while ((pos = strstr(shader_src, "WORKGROUP_SIZE_X")) != NULL) {
        sprintf(pos, "%-15d", WORKGROUP_SIZE_X);
        *(pos + 15) = ' ';
    }
    while ((pos = strstr(shader_src, "WORKGROUP_SIZE_Y")) != NULL) {
        sprintf(pos, "%-15d", WORKGROUP_SIZE_Y);
        *(pos + 15) = ' ';
    }
    while ((pos = strstr(shader_src, "WORKGROUP_SIZE_Z")) != NULL) {
        sprintf(pos, "%-15d", WORKGROUP_SIZE_Z);
        *(pos + 15) = ' ';
    }

    while ((pos = strstr(shader_src, "USE_VARIABLE_GROUP_SIZE")) != NULL) {
        sprintf(pos, "%-22d", variable_group_size ? 1 : 0);
        *(pos + 22) = ' ';
    }

    // mesa doesn't support KHR_shader_subgroup in GL
    if (0) {
        while ((pos = strstr(shader_src, "USE_SUBGROUPS")) != NULL) {
            sprintf(pos, "%-12d", 1);
            *(pos + 12) = ' ';
        }
    }

    if (0)
        printf("%s\n", shader_src);

    GLenum err;
    const char *const_shader_src = shader_src;
    glShaderSource(shader, 1, &const_shader_src, NULL);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "glShaderSource: 0x%x\n", err);
        exit(2);
    }

    free(shader_src);
    shader_src = NULL;

    glCompileShader(shader);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        char b[4096];
        GLsizei l;
        glGetShaderInfoLog(shader, sizeof(b), &l, b);
        fprintf(stderr, "glCompileShader: %s\n", b);
        exit(2);
    }

    int compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    assert(glGetError() == GL_NO_ERROR);
    if (compiled != GL_TRUE) {
        char b[4096];
        GLsizei l;
        glGetShaderInfoLog(shader, sizeof(b), &l, b);
        fprintf(stderr, "GL_COMPILE_STATUS: %s\n", b);
        exit(2);
    }

    assert(compiled == GL_TRUE);

    GLuint prog = glCreateProgram();

    glAttachShader(prog, shader);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "glAttachShader: 0x%x\n", err);
        exit(2);
    }

    glLinkProgram(prog);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        char b[4096];
        GLsizei l;
        glGetProgramInfoLog(prog, sizeof(b), &l, b);
        fprintf(stderr, "glLinkProgram: %s\n", b);
        exit(2);
    }
    int linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    assert(glGetError() == GL_NO_ERROR);
    if (linked != GL_TRUE) {
        char b[4096];
        GLsizei l;
        glGetProgramInfoLog(shader, sizeof(b), &l, b);
        fprintf(stderr, "GL_LINK_STATUS: %s\n", b);
        exit(2);
    }

    glUseProgram(prog);
    err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "glUseProgram: 0x%x\n", err);

        char b[4096];
        GLsizei l;
        glGetProgramInfoLog(prog, sizeof(b), &l, b);
        fprintf(stderr, "%s\n", b);
        exit(2);
    }

    struct timespec start, end;

    if (perf.enabled) {
        // perf.dbg = true;

        perf.compute_metrics_basic.off_thread_occupancy_pct = 0;
        perf.compute_metrics_basic.off_threads = 0;
        perf.compute_metrics_basic.off_time_ns = 0;
        perf.pipeline_statistics.off_cs_invocations = 0;

        char qname0[] = "Compute Metrics Basic Gen9";
        char qname1[] = "Pipeline Statistics Registers";
        query_query(qname0, false);
        query_query(qname1, true);

        int err;

        do {
            glBeginPerfQueryINTEL(perf.compute_metrics_basic.queryHandle);
            err = glGetError();
            if (err == GL_INVALID_OPERATION)
                usleep(10000);
        } while (err == GL_INVALID_OPERATION);
        assert(err == GL_NO_ERROR);

        do {
            glBeginPerfQueryINTEL(perf.pipeline_statistics.queryHandle);
            err = glGetError();
            if (err == GL_INVALID_OPERATION)
                usleep(10000);
        } while (err == GL_INVALID_OPERATION);
        assert(err == GL_NO_ERROR);

        if (clock_gettime(CLOCK_MONOTONIC, &start))
            abort();
    }

    GLuint num_groups_x = (GLuint)ceil(WIDTH / (float)WORKGROUP_SIZE_X);
    GLuint num_groups_y = (GLuint)ceil(HEIGHT / (float)WORKGROUP_SIZE_Y);
    GLuint num_groups_z = (GLuint)ceil(DEPTH / (float)WORKGROUP_SIZE_Z);

    if (variable_group_size) {
        glDispatchComputeGroupSizeARB(num_groups_x, num_groups_y, num_groups_z,
                WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z);
    } else {
        glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    }
    err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "glDispatchCompute: 0x%x\n", err);
        exit(2);
    }

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    assert(glGetError() == GL_NO_ERROR);

    glFinish();
    assert(glGetError() == GL_NO_ERROR);

    if (perf.enabled) {
        if (clock_gettime(CLOCK_MONOTONIC, &end))
            abort();

        glEndPerfQueryINTEL(perf.pipeline_statistics.queryHandle);
        assert(glGetError() == GL_NO_ERROR);

        glEndPerfQueryINTEL(perf.compute_metrics_basic.queryHandle);
        assert(glGetError() == GL_NO_ERROR);

        uint bytesWritten = 0;

        char *cmb_queryData = malloc(perf.compute_metrics_basic.dataSize);
        char *ps_queryData = malloc(perf.pipeline_statistics.dataSize);

        glGetPerfQueryDataINTEL(perf.compute_metrics_basic.queryHandle,
                GL_PERFQUERY_WAIT_INTEL, perf.compute_metrics_basic.dataSize,
                cmb_queryData, &bytesWritten);
        assert(glGetError() == GL_NO_ERROR);
        if (bytesWritten != perf.compute_metrics_basic.dataSize)
            abort();

        glGetPerfQueryDataINTEL(perf.pipeline_statistics.queryHandle,
                GL_PERFQUERY_WAIT_INTEL, perf.pipeline_statistics.dataSize,
                ps_queryData, &bytesWritten);
        assert(glGetError() == GL_NO_ERROR);
        if (bytesWritten != perf.pipeline_statistics.dataSize)
            abort();

        if (perf.dbg) {
            printf("CMB:\n");
            for (unsigned i = 0; i < perf.compute_metrics_basic.dataSize / 8; ++i)
                printf("%u %lu\n", i * 8, *(uint64_t *)(cmb_queryData + i * 8));
            printf("PS:\n");
            for (unsigned i = 0; i < perf.pipeline_statistics.dataSize / 8; ++i)
                printf("%u %lu\n", i * 8, *(uint64_t *)(ps_queryData + i * 8));
        }

        uint64_t threads = 0;
        uint64_t time_ns = 0;
        float thread_occupancy_pct = 0;
        uint64_t cs_invocations = 0;

        if (perf.compute_metrics_basic.off_threads)
            threads = *(uint64_t *)(cmb_queryData + perf.compute_metrics_basic.off_threads);

        if (perf.compute_metrics_basic.off_time_ns)
            time_ns = *(uint64_t *)(cmb_queryData + perf.compute_metrics_basic.off_time_ns);

        if (perf.compute_metrics_basic.off_thread_occupancy_pct)
            thread_occupancy_pct = *(float *)(cmb_queryData + perf.compute_metrics_basic.off_thread_occupancy_pct);

        if (perf.pipeline_statistics.off_cs_invocations)
            cs_invocations = *(uint64_t *)(ps_queryData + perf.pipeline_statistics.off_cs_invocations);

        uint64_t cpu_time_ns = 1000ULL * 1000 * 1000 * (end.tv_sec - start.tv_sec) +
                end.tv_nsec - start.tv_nsec;

        if (perf.show_csv) {
            fprintf(perf.statsFile, "%d,%d,%d,", WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z);
            fprintf(perf.statsFile, "%lu,", time_ns);
            fprintf(perf.statsFile, "%lu,", threads);
            fprintf(perf.statsFile, "%lu,", cs_invocations);
            fprintf(perf.statsFile, "%lu,", threads ? cs_invocations / threads : 0);
            fprintf(perf.statsFile, "%d,", (int)thread_occupancy_pct);
            fprintf(perf.statsFile, "%lu\n", cpu_time_ns);
        } else {
            printf("EU Thread Occupancy:   %f %%\n", thread_occupancy_pct);
            printf("CS Threads Dispatched: %lu\n", threads);
            printf("GPU Time Elapsed:      %lu ns\n", time_ns);
            printf("CS Invocations:        %lu\n", cs_invocations);
            printf("CPU Time Elapsed:      %lu ns\n", cpu_time_ns);
        }

        free(cmb_queryData);
        free(ps_queryData);

        glDeletePerfQueryINTEL(perf.compute_metrics_basic.queryHandle);
        assert(glGetError() == GL_NO_ERROR);
        glDeletePerfQueryINTEL(perf.pipeline_statistics.queryHandle);
        assert(glGetError() == GL_NO_ERROR);
    }

    struct Pixel *result = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (!result) {
        fprintf(stderr, "glMapBuffer: 0x%x\n", glGetError());
        exit(2);
    }

    save_data(result, WIDTH, HEIGHT, DEPTH);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glDeleteShader(shader);
    glDeleteProgram(prog);
    eglDestroyContext(disp, ctx);
    eglTerminate(disp);
    gbm_device_destroy(gbm);
    close(fd);
    return 0;
}
