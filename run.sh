#!/bin/bash
D=10
total_repeats=1000
repeats=$((total_repeats / D))
for i in $(seq 1 $D)
do
    python dev/main.py -s all -a lil-ucb -r $repeats -t 40 -g 100 -z 0.01 -u 20 -m 30 -c 1.0 -e single-debias -x 0.5 -b 1.0 -w 0.1 -k $i -p -y -o out -v
done

python dev/process_div.py -i out -a lil-ucb -x 0.5 -b 1.0 -s all -t 40 -w 0.1 -e single-debias -d gauss -k $D -r $total_repeats 
