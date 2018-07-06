#__precompile__()

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
