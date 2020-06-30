#!/bin/bash -e

rm -f mandelbrot.png
~/glsllang/bin/glslangValidator -V shaders/shader.comp -o shaders/comp.spv  && make && time sh -c "mygl.sh ./vulkan_minimal_compute" && xdg-open mandelbrot.png
