#define GL_GLEXT_PROTOTYPES

#include <assert.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <fcntl.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <gbm.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "shared.h"

static int WIDTH;
static int HEIGHT;
static int DEPTH;

static int WORKGROUP_SIZE_X;
static int WORKGROUP_SIZE_Y;
static int WORKGROUP_SIZE_Z;

int
main(int argc, char *argv[])
{
    if (argc != 9) {
        fprintf(stderr, "not enough arguments\n");
        exit(2);
    }

    WIDTH = atoi(argv[3]);
    HEIGHT = atoi(argv[4]);
    DEPTH = atoi(argv[5]);
    WORKGROUP_SIZE_X = atoi(argv[6]);
    WORKGROUP_SIZE_Y = atoi(argv[7]);
    WORKGROUP_SIZE_Z = atoi(argv[8]);

    if (WORKGROUP_SIZE_X == 0 || WORKGROUP_SIZE_Y == 0 || WORKGROUP_SIZE_Z == 0||
            WIDTH == 0 || HEIGHT == 0 || DEPTH == 0)
        abort();

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
    struct Pixel *buffer = NULL;//malloc(bufferSize);
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    assert(glGetError() == GL_NO_ERROR);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    assert(glGetError() == GL_NO_ERROR);

    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, buffer, GL_STATIC_READ);
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

    glDispatchCompute(
            (uint32_t)ceil(WIDTH / (float)WORKGROUP_SIZE_X),
            (uint32_t)ceil(HEIGHT / (float)WORKGROUP_SIZE_Y),
            (uint32_t)ceil(DEPTH / (float)WORKGROUP_SIZE_Z));
    err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "glDispatchCompute: 0x%x\n", err);
        exit(2);
    }

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    assert(glGetError() == GL_NO_ERROR);

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
