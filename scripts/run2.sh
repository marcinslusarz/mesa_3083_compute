#!/bin/bash -e

cat shaders/shader.comp | sed "s/WORKGROUP_SIZE_X 32/WORKGROUP_SIZE_X $1/" | sed "s/WORKGROUP_SIZE_Y 32/WORKGROUP_SIZE_Y $2/" > tmp.comp
~/glslang/bin/glslangValidator --target-env vulkan1.2 -V tmp.comp -o shaders/comp.spv

rm -f mandelbrot.png
GROUP_X=$1 GROUP_Y=$2 CSV=1 mygl.sh $GDB ./vulkan_minimal_compute
if [ ! -f mandelbrot.png ]; then
	echo "output file doesn't exist"
	exit 1
fi
