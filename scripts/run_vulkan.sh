#!/bin/bash -e

~/glslang/bin/glslangValidator -DUSE_SUBGROUPS=1 -DWIDTH=$2 -DHEIGHT=$3 -DDEPTH=$4 -DWORKGROUP_SIZE_X=$5 -DWORKGROUP_SIZE_Y=$6 -DWORKGROUP_SIZE_Z=$7 --target-env vulkan1.2 -V $1 -o shaders/comp.spv --quiet

rm -f result.png stats.csv data.csv
ANV_ENABLE_PIPELINE_CACHE=0 CSV=1 mygl.sh $GDB ./vulkan_compute $2 $3 $4 $5 $6 $7
if [ ! -f result.png ]; then
	echo "output file doesn't exist"
	exit 1
fi
