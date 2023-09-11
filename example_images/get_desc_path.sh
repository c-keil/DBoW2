#!/bin/bash

#read paths from file
paths=()
readarray -t paths  < $1

rm -f $2

for p in ${paths[@]}
do
fname=$(basename $p)
echo $(dirname $p)/irpoint_v5/descriptors/${fname/.png/.npy} >> $2
# echo $(dirname $p)/sp_v6/descriptors/${fname/.png/.npy} >> $2
done