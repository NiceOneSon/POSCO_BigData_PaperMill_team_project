#!/bin/bash

for i in 5 10 20 30
do
  for j in 10 30 60 120
  do
    for k in 30 60 120 240
    do
        python auto_paper.py $i $j $k 
    done
  done
done