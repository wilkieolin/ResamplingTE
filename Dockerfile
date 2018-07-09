# start off with the standard data science notebook from jupyter,
# as it includes the kernels and setup we need.

FROM jupyter/datascience-notebook

USER root

RUN julia -e 'Pkg.add("PyPlot")' && \
  julia -e 'Pkg.add("MAT")' && \
  julia -e 'Pkg.add("StatsBase")'

# add the source files into the jupyter working directory
ADD . /home/jayvan/work
