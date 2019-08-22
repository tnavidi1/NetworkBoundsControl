#!/bin/bash

for i in {0..9}
do

python makeDemand.py --solarPen=$i --storagePen=0

echo made demand files

python pypowerTest.py --Qcap=-0.79 --solarPen=$i --storagePen=0

echo pypower ran

python SVR_code.py --train=1 --solarPen=$i --storagePen=0

echo SVR_code trained

python SVR_code.py --train=0 --solarPen=$i --storagePen=0 >> SVR_test_results-1.txt

done