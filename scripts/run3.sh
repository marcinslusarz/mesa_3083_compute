#!/bin/bash -e

echo "x:int,y:int,z:int,time_ms:int,threads:int,invocations:int,simd:int,thread_occupancy_pct:int" | tee runtime.csv

for x in 1 2 4 8 16 32 64 128 256 512; do
	for y in 1 2 4 8 16 32 64 128 256 512; do
		sz=$(($x * $y))
		if [ $sz -le 1792 ]; then
			./run2.sh $1 $2 $3 $x $y | grep -v -e "$1" -e "x:int" | tee -a runtime.csv
			cp data.csv data_${2}x${3}_${x}x${y}.csv
		fi
	done
done
