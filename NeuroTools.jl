#=
Copyright 2018 Wilkie Olin-Ammentorp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

module NeuroTools

  using MAT, PyPlot, StatsBase, HDF5

  import Base.length, Base.convert, Base.Iterators.flatten

  export
  #Types
  Recording,
  MEARecording,
  StimMEARecording,
  JointDist,
  length,
  chop_rec,
  islice,
  tslice,

  #Data IO
  loadRecording,
  loadBrianFile,
  loadBrianCulture,
  findProcFiles,
  procFileInfo,
  loadProcFile,
  stimIndices,
  convert,

  #Data analysis
  spikeRandomization,
  spikeDither,
  cwShuffle,
  MCMC,
  joint_dist_resample,
  TE_resample,
  MDS,
  findCoincident,
  crossCorrelogram,
  crossCorrelate,
  full_CC,
  permutation_test,
  filteredCorrelation,
  surrogateDistribution,
  stimulation_entropy,
  transfer_entropy,
  sp_joint_dist,
  self_prediction,
  reverse_marginal_dist,
  te_calc,
  count_vector,
  surr_TE,
  surr_SES,
  MCMC_TE,
  full_TE,
  resampling_TE,

  #Results analysis
  u_test,
  edge_filter,
  edge_count,
  mprecision,
  maccuracy,
  mrecall,
  mspecificity,
  f1,
  centricesByBatch,
  betweennessCentralities,
  degreeCentralities,
  pca,
  emergence,
  permanent_emergence,

  #Plotting
  complot,
  ecdf,
  pcaPlot,
  pcaPlotByDays,
  pcaPlotVsDays,
  plotLoadings,
  plotNormPMF,
  delay_plot,
  connectionMap

  #source files
  include("src/types.jl")
  include("src/dataio.jl")
  include("src/transfer_entropy.jl")
  include("src/crosscorr.jl")
  include("src/analysisfunctions.jl")
  include("src/plotting.jl")

end
