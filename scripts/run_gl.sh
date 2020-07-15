#!/bin/bash -e

rm -f result.png stats.csv data.csv
MESA_GLSL_CACHE_DISABLE=1 CSV=1 mygl.sh $GDB ./gl_compute $1 $2 $3 $4 $5 $6 $7 $8
if [ ! -f result.png ]; then
	echo "output file doesn't exist"
	exit 1
fi
