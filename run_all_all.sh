#!/bin/bash

# interrupted at i =

for i in {1..9}
do

# python makeDemand.py --solarPen=6 --storagePen=0
# echo made demand files

python EV_Profiles.py --evPen=$i
echo made EV profiles

# python pypowerTest.py --Qcap=-0.79 --solarPen=6 --storagePen=0 --final=0
# echo pypower SVR training data ran

# python SVR_code.py --train=1 --solarPen=6 --storagePen=0
# echo SVR_code trained

python BoundsOptimization.py --solarPen=6 --storagePen=0 --evPen=$i --bounds=0

# python BoundsOptimization.py --solarPen=6 --storagePen=0 --evPen=$i --bounds=1

done

for i in {0..9}
do

python makeDemand.py --solarPen=$i --storagePen=0
echo made demand files

# python EV_Profiles.py --evPen=1
# echo made EV profiles

python pypowerTest.py --Qcap=-0.79 --solarPen=$i --storagePen=0 --final=0
echo pypower SVR training data ran

python SVR_code.py --train=1 --solarPen=$i --storagePen=0
echo SVR_code trained

# Not necessary to run as results are contained in SVR training data
python BoundsOptimization.py --solarPen=$i --storagePen=0 --evPen=1 --bounds=0

# python BoundsOptimization.py --solarPen=$i --storagePen=0 --evPen=1 --bounds=1

done

for i in {1..9}
do

python makeDemand.py --solarPen=6 --storagePen=$i
echo made demand files

# python EV_Profiles.py --evPen=4
# echo made EV profiles

# python pypowerTest.py --Qcap=-0.79 --solarPen=6 --storagePen=0 --final=0
# echo pypower SVR training data ran

# python SVR_code.py --train=1 --solarPen=6 --storagePen=0
# echo SVR_code trained

python BoundsOptimization.py --solarPen=6 --storagePen=$i --evPen=4 --bounds=0

python BoundsOptimization.py --solarPen=6 --storagePen=$i --evPen=4 --bounds=1

done
