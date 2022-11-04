# ~\~ language=Gnuplot filename=scripts/plot-sigmoid.gnuplot
# ~\~ begin <<README.md|scripts/plot-sigmoid.gnuplot>>[init]
set term svg
set xrange [-5:5]
set yrange [-1.1:1.2]
set key right top opaque box
plot tanh(x) t'tanh(x)'
# ~\~ end
