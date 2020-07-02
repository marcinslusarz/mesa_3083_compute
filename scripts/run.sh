#!/bin/bash -e

rm -f mandelbrot.png
~/glsllang/bin/glslangValidator -DWORKGROUP_SIZE_X=$4 -DWORKGROUP_SIZE_Y=$5 -DWIDTH=$2 -DHEIGHT=$3 -target-env vulkan1.2 -V $1 -o shaders/comp.spv
make
ANV_ENABLE_PIPELINE_CACHE=0 MESA_GLSL_CACHE_DISABLE=1 $GDB mygl.sh ./vulkan_minimal_compute && xdg-open mandelbrot.png
