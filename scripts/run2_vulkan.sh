#!/bin/bash -e

echo "x:int,y:int,z:int,time_ms:int,threads:int,invocations:int,simd:int,thread_occupancy_pct:int" | tee runtime.csv

for x in 1 2 4 8 16 32 64 128 256 512; do
	for y in 1 2 4 8 16 32 64 128 256 512; do
		for z in 1 2 4 8 16 32 64; do
			sz=$(($x * $y * $z))
			if [ $sz -le 1792 ]; then
				./run_vulkan.sh $1 $2 $3 $4 $x $y $z
				cat stats.csv | csv-header -m | tee -a runtime.csv
				cp data.csv data_${2}x${3}x${4}_${x}x${y}x${z}.csv
			fi
		done
	done
done
