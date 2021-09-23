for i in 8 16 32 64 128
do
	for rej in low high
	do
		for ty in range dist
		do
			x="grep $ty.*i$i.*$rej benches/unif_bench_logs"
			echo $x
			$x | sort -k10n
			echo ""
		done
	done
done
