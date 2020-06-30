#!/bin/bash -e

for x in 1 2 4 8 16 32 64 128 256 512; do
	for y in 1 2 4 8 16 32 64 128 256 512; do
		sz=$(($x * $y))
		if [ $sz -le 1792 ]; then
			./run2.sh $x $y
		fi
	done
done
