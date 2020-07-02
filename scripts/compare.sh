#!/bin/bash -e

csv-merge -N before -p "$1" -N after -p "$2" |
csv-sqlite -T \
	"select before.x,
		before.y,
		before.time_ms as time_ms_before,
		after. time_ms as time_ms_after,
		printf('%.2f', 1.0 * before.time_ms / after.time_ms) as speedup,
		before.threads as threads_before,
		after. threads as threads_after,
		before.invocations as invocations_before,
		after. invocations as invocations_after,
		before.simd as simd_before,
		after. simd as simd_after
	   from before, after
	  where before.x = after.x
	    and before.y = after.y
	    and before.z = after.z
	    and (   before.simd        != after.simd
	         or before.threads     != after.threads
		 or before.invocations != after.invocations
		 or (before.x == 4 and before.y == 4)
		)
	  order by 1.0 * before.time_ms / after.time_ms desc" -s
