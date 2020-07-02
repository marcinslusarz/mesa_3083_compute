#!/bin/bash -e

rm -f mandelbrot.png
~/glsllang/bin/glslangValidator -DWORKGROUP_SIZE_X=$4 -DWORKGROUP_SIZE_Y=$5 -DWIDTH=$2 -DHEIGHT=$3 -target-env vulkan1.2 -V $1 -o shaders/comp.spv && make && $GDB mygl.sh ./vulkan_minimal_compute && xdg-open mandelbrot.png
