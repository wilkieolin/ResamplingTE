masterpath = "/home/wilkie/jupyter/neuroTools2/outernew2.mat"

"""
    findProcFiles(batches=collect(1:6),modes=[], cultures=[], days=[], densities=[])

Returns a list of procfile recordings which match given conditions.
Must give the recording mode (1=spontaneous, 2=stimulated) and a batch number.
Specific cultures and days are additional keyword options.
"""
function findProcFiles(batches::Array=collect(1:6); modes::Array=[], cultures::Array=[], days::Array=[], densities::Array=[])
    list = matread(masterpath)
    list = list["list"]

    #give only summary files (not full binary)
    all_inds = find(list[:,6].==1.0)

    #find the files which match the requested mode
    if !isempty(modes)
        modes = convert(Array{Any,1},modes)

        #if we have a string arg for mode, convert it to numerical
        types = typeof.(modes)
        for i in find(types.==String)
            word = modes[i]

            #look for the words describing mode
            if word == "spont" || word == "spontaneous"
                modes[i] = 1.0
            elseif word == "stim" || word == "stimulated"
                modes[i] = 2.0
            else
                println("Mode '",word,"' not recognized.")
            end
        end

        mode_inds = Int64[]
        #find files which are stimulated/spont
        for mode in modes
            mode_inds = union(mode_inds, find(mode .== list[:,1]))
        end

        all_inds = intersect(all_inds, mode_inds)
    end

    #find the files which match the requested density
    if !isempty(densities)
        densities = convert(Array{Any,1},densities)

        #if we have a string arg for density, convert it to numerical
        types = typeof.(densities)
        for i in find(types.==String)
            word = densities[i]

            #look for the words describing mode
            if word == "dense"
                densities[i] = 1.0
            elseif word == "small"
                densities[i] = 2.0
            elseif word == "sparse"
                densities[i] = 3.0
            elseif word == "small&sparse"
                densities[i] = 4.0
            elseif word == "ultrasparse"
                densities[i] = 5.0
            else
                println("Density '",word,"' not recognized.")
            end

        end

        density_inds = Int64[]
        #find files which are stimulated/spont
        for density in densities
            density_inds = union(density_inds, find(density .== list[:,2]))
        end

        all_inds = intersect(all_inds, density_inds)

    end

    if !isempty(batches)
        batch_inds = Int64[]
        #find the right indices for the batches
        for batch in batches
            batch_inds = union(batch_inds, find(batch .== list[:,3]))
        end

        all_inds = intersect(all_inds, batch_inds)
    end

    if !isempty(cultures)
        culture_inds = Int64[]
        #find the right indices for the cultures
        for culture in cultures
            culture_inds = union(culture_inds, find(culture .== list[:,4]))
        end

        all_inds = intersect(all_inds, culture_inds)
    end

    if !isempty(days)
        day_inds = Int64[]
        #find the right indices for the days
        for day in days
            day_inds = union(day_inds, find(day .== list[:,5]))
        end

        all_inds = intersect(all_inds, day_inds)
    end

    return all_inds


end

"""
    function findProcFilesPairs(batches::Array=collect(1:6); modes::Array=[], cultures::Array=[], days::Array=[], densities::Array=[])

Given batch/culture/day information, find all the pairs of files which
have both a stimulated and a spontaneous recording.
"""
function findProcFilesPairs(batches::Array=collect(1:6); cultures::Array=[], days::Array=[], densities::Array=[])
    #find the spontaneous recordings in a batch
    spont_indices  = findProcFiles(batches, modes=[1.0], cultures=cultures, days=days, densities=densities)

    #create a dictionary for pairs to go into
    pairs = Dict{Int, Array{Int,1}}()

    #for each spontaneous file, find if it has a matching stimulated file.
    #if so, put it into the dictionary.
    for i = 1:size(spont_indices,1)
        #find a matching stim file
        info = procFileInfo(spont_indices[i])
        day = [info["recording day"]]
        culture = [info["culture"]]
        batch = [info["batch"]]
        density = [info["density"]]

        stim_index = findProcFiles(batch,modes=[2.0],cultures=culture,densities=density,days=day)

        #does it exist?
        if !isempty(stim_index)
            pairs[spont_indices[i]] = stim_index
        end
    end

    return pairs
end

"""
    function procFileInfo(procfiles::Array{Int64,1}, param::String)

Returns an array of the int-valued parameters requested for a list of procfiles.
Recognizes mode, density, culture, batch, day, and recording mode.
"""
function procFileInfo(procfiles::Array{Int64,1}, param::String)
    column_decoder = Dict{String,Int}()
    column_decoder["mode"] = 1
    column_decoder["density"] = 2
    column_decoder["batch"] = 3
    column_decoder["culture"] = 4
    column_decoder["day"] = 5
    column_decoder["recording mode"] = 6

    col = column_decoder[param]

    list = matread(masterpath)
    list = list["list"][procfiles,:]

    return list[:,col]

end


"""
    procFileInfo(index::Int64)

Returns a dictionary of the information for a given recording.
"""
function procFileInfo(procfile::Int)
    checkIndex(procfile)

    #load the procfiles list
    list = matread(masterpath)
    list = list["list"][procfile,:]

    #create the dict for the information to go into
    info = Dict{String, Any}()

    #is it spontaneous, or stimulated?
    if list[1] == 1
        info["mode"] = "spontaneous"
    else
        info["mode"] = "stimulated"
    end

    #what is the plating density?
    if list[2] == 1
        info["density"] = "dense"
    elseif list[2] == 2
        info["density"] = "small"
    elseif list[2] == 3
        info["density"] = "sparse"
    elseif list[2] == 4
        info["density"] = "small&sparse"
    elseif list[2] == 5
        info["density"] = "ultrasparse"
    end

    #batch and culture numbers
    info["batch"] = list[3]
    info["culture"] = list[4]
    info["recording day"] = list[5]

    #storage style
    if list[6] == 1
        info["storage style"] = "summary matlab"
    else
        info["storage style"] = "full binary"
    end

    #recording period
    if list[7] == 1
        info["recording period"] = "regular daytime"
    else
        info["recording period"] = "longer overnight"
    end

    return info
end

"""
    function tryProcFile(index::Int64)

See if a procfile can be opened.
"""
function tryProcFile(index::Int64)
    #nicely check to see if a procfile recording can be opened
    checkIndex(index)

    try
      file = matopen(string("procfiles/innernew",index,".mat"))
    catch
      println(string("Error with procfile ", index))
      return false
    end

    return true
end

"""
    loadProcFile(index::Int64)

Returns a Recording. Opens the procfile of the given index.
"""
function loadProcFile(index::Int64; dir::String="/media/hgst/procfiles2/procfiles/")
  #load a recording from a procfile
  #use only for Potter files with 61-stimulus index
  checkIndex(index)

  # if !isdir("procfiles
  #     throw(ArgumentError("Procfiles directory not found"))
  # end

  file = matread(string(dir,"innernew",index,".mat"))

  #the spk key contains the measured spiking information for all recordings
  vars = file["spk"]

  channels = convert(Array{Int,2},vars["chs"])
  channels = channels[:,1]
  #PROCFILES ARE ALREADY 1-INDEXED!!

  #general configuration detection
  #find the highest electrode number present, to tell if it's stimulated or spontaneous
  numElectrodes = extrema(channels)[2]

  #this is a potter recording, #electrodes = 60
  #find if there are dead electrodes (no spikes on channels in range)
  electrodes = collect(1:60)
  deadElectrodes = setdiff(electrodes,unique(channels))
  #set the active electrodes
  activeElectrodes = setdiff(electrodes,deadElectrodes)

  #load the times
  times = vars["tms"]
  times = times[:,1]

  #set up the detected configuration
  config = Dict{String,Any}()
  config["channels"] = 60
  config["dead electrodes"] = deadElectrodes
  config["ground electrode"] = 15
  config["active electrodes"] = activeElectrodes
  config["recording number"] = index
  config["delta_t"] = 5e-5

  config["info"] = procFileInfo(index)

  #find if it's a stimulated file
  if numElectrodes == 61

      #store the times at which a stimulus occurred
      stimTimes = file["tri"]["tms"][:,1]
      #store the channel which was stimulated
      stimChannels = convert(Array{Int64,1},file["tri"]["stm"][:,1])

      #check and see if there were any rounds of stimulation which were skipped
      triZeroInds = find(stimChannels .== 0)
      #if there are, remove those rounds since we don't know what they did...
      if size(triZeroInds,1) != 0
        #filter the zero values from the array
        filter!(x -> x !== 0, stimChannels)

        #filter the corresponding spike times out of the other array
        triZeroTimes = stimTimes[triZeroInds]
        filter!(x -> !(x in triZeroTimes), stimTimes)
      end

      #find the indices in the spk data where stimuli occurred
      isStimIndex = [times[i] in stimTimes for i = 1:size(times,1)]
      stimIndices = find(isStimIndex)

      #find the stimulated channels
      config["stimulated electrodes"] = unique(stimChannels)

      #store the indices which point to the next event after the stimulus occurred
      #shift the stimIndices to compensate for the elements ahead which will be removed
      stimIndices = stimIndices - collect(0:size(stimIndices,1)-1)

      #invert the condition, and pull the stimuli from the actual recording
      times = times[find(.!(isStimIndex))]
      channels = channels[find(.!(isStimIndex))]

      #and lastly, remove any orphaned channel 61 stimuli in the spk data which correspond to the 0 channel stimulus in tri.
      spkNonZeroInds = find(channels .!= 61)
      channels = channels[spkNonZeroInds]
      times = times[spkNonZeroInds]

      #check to see if a stimulus occurs as the first recorded feature, and remove it if so
      #there can be no 'before' period for this stimulus, so it isn't useful
      if (stimIndices[1] == 1)
        print("Removing isolated start stimulus")
        stimIndices = stimIndices[2:end]
      end

      #check to see if a stimulus occurs as the last recorded feature, and remove it if so
      #there can be no 'after' period for this stimulus, so it isn't useful
      if (stimIndices[end] > size(channels,1))
        print("Removing isolated end stimulus")
        stimIndices = stimIndices[1:end-1]
      end

      recording = StimMEARecording(channels, times, stimTimes, stimIndices, stimChannels, config)
  else
      #it's just spontaneous
      recording = MEARecording(channels, times, config)
  end

  return recording
end

"""
    function loadBrianFile(filename::AbstractString)

Given an HDF5 file at "filename", attempt to open it and load it into a recording.
"""
function loadBrianFile(filename::AbstractString)
    #open the file
    spikeFile = h5open(filename, "r")
    #check to see if it has the appropriate datasets for a spiketrain file
    if !(exists(spikeFile, "chs")) && !(exists(spikeFile, "tms"))
        println("Not a spike file, data missing")
        return
    end
    #convert the Float32 python indices into Int64 Julia ints
    channels = convert(Array{Int,1}, spikeFile["chs"][:])
    #one-index the channels
    channels += 1

    #convert Float32s to Float64s
    times = convert(Array{Float64,1}, spikeFile["tms"][:])

    #general configuration detection
    #find the highest electrode number present, and set as the number of electrodes
    numElectrodes = extrema(channels)[2]
    #find if there are dead electrodes (no spikes on channels in range)
    electrodes = collect(1:numElectrodes)

    config = Dict{String,Any}()
    config["channels"] = numElectrodes

    recording = MEARecording(channels, times, config)

    return recording
end

"""
Given a recording which was stimulated, find the indices of events which occured nearest to the stimuli in the recording.
"""
function stimIndices(stim_times::Array{<:Real,1}, times::Array{<:Real,1})
    #find the indices after stimulation events
    stimuli = length(stim_times)
    events = length(times)
    stim_inds = zeros(Int,stimuli)
    i = Int(1)
    for stimulus in 1:stimuli
        #what is the time this stimulation occured?
        stim_time = stim_times[stimulus]
        while i < events
            #is the current event index after this time?
            if times[i] > stim_time
                break
            else
                i += 1
            end
        end
        #if so, store it in the array
        stim_inds[stimulus] = i
    end

    return stim_inds
end

function loadBrianCulture(filename::AbstractString)
    #open the file
    spikeFile = h5open(filename, "r")
    #check to see if it has the appropriate datasets for a spiketrain file
    if !(exists(spikeFile, "chs")) && !(exists(spikeFile, "tms"))
        println("Not a spike file, data missing")
        return
    end
    #convert the Float32 python indices into Int64 Julia ints
    channels = convert(Array{Int,1}, spikeFile["chs"][:])
    #one-index the channels
    channels += 1

    #convert Float32s to Float64s
    times = convert(Array{Float64,1}, spikeFile["tms"][:])

    #general configuration detection
    #find the highest electrode number present, and set as the number of electrodes
    numElectrodes = extrema(channels)[2]
    electrodes = collect(1:numElectrodes)

    config = Dict{String,Any}()
    config["channels"] = numElectrodes

    #detect if there has been a stimulation
    if exists(spikeFile, "t_stim") && exists(spikeFile, "i_stim")
        stim_times = convert(Array{Float64,1},spikeFile["t_stim"][:])
        stim_channels = convert(Array{Int64,1},spikeFile["i_stim"][:])

        stim_indices = stimIndices(stim_times, times)

        recording = StimMEARecording(channels, times, stim_times, stim_indices, stim_channels, config)
    #otherwise, we have a regular non-stimulated recording (spontaneous)
    else
        recording = MEARecording(channels, times, config)
    end

    return (recording, spikeFile["adj"][:,:])
end
