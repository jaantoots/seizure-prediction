set terminal postscript eps enhanced color background "white"

lossf=sprintf("%s/loss.log", ARG1)
validatef=sprintf("%s/validate.clean.log", ARG1)
set output sprintf("%s/score.eps", ARG1)

set multiplot layout 2,1

set ylabel "Loss"
plot lossf u 1:2 every ::1 w l notitle

set ylabel "Validate"
plot validatef u 1:2 w lp notitle

unset multiplot
set output
