"""
Shuffle a recording channel-wise, preserving each channel's ISI distribution
"""
function cwShuffle(recording::Recording)
    n = length(recording)
    times = zeros(Float64,n)
    channels = zeros(Int64,n)
    t0 = recording.times[1]
    dim = recording.config["channels"]

    s = 1
    for i in 1:dim
        #which indices in the recording correspond to spikes on this channel?
        inds = find(x->x==i, recording.channels)
        #skip dead channels
        if isempty(inds)
            continue
        end
        spikes = length(inds)
        #how far from the start of the recording is the first spike in this channel?
        offset = recording.times[inds[1]] - t0
        #calculate and shuffle the interspike intervals for this channel
        isi = shuffle!(diff(recording.times[inds]))
        #add the times relative to the offset to calculate the surrogate spike times
        new_ts = cumsum(cat(1,offset,isi))
        #assign these values into the array
        times[s:s+spikes-1] = new_ts
        channels[s:s+spikes-1] = i
        #update the start position in the array for the next channel
        s += spikes
    end

    #now sort the recording
    perm = sortperm(times)
    times = times[perm]
    channels = channels[perm]

    return MEARecording(channels,times,recording.config)
end

"""
Dither a recording's spike times - changes times by uniform sample within +/- tw
"""
function spikeDither(recording::Recording, tw::Real)
    n = length(recording)
    times = copy(recording.times)
    for i in 1:n
        times[i] += (rand() - 0.5) * 2 * tw
    end

    return MEARecording(recording.channels, times, recording.config)
end


"""
Find the subset of spikes within the recording which happen at identical times.
Returns an array of indices which indicate which spikes are coincident.
"""
function findCoincident(times::Array{Float64,1})
  length = size(times,1)

  isCoincidentSpike = zeros(length)
  #find the pairs which have no time-difference between them.
  zerodiffs = find(diff(times).==0)

  #add 1 to the first spike in the pair.
  isCoincidentSpike[zerodiffs] += 1
  #add 1 to the second spike in the pair.
  isCoincidentSpike[zerodiffs.+1] += 1

  #return the indices where the sum is greater than zero; these spikes are coincident.
  return find(isCoincidentSpike.>0)
end


"""
Finds and returns the indices of spikes which are not time-coincident.
Used by spike pairs methods to make a cleaner analysis.
"""
function findNotCoincident(times::Array{Float64,1})
    length = size(times,1)

    isCoincidentSpike = zeros(length)
    #find the pairs which have no time-difference between them.
    zerodiffs = find(diff(times).==0)

    #add 1 to the first spike in the pair.
    isCoincidentSpike[zerodiffs] += 1
    #add 1 to the second spike in the pair.
    isCoincidentSpike[zerodiffs.+1] += 1

    #return the indices where the sum is zero; these spikes are non-coincident.
    return find(isCoincidentSpike.==0)
end

"""
Carry out the cross-correlation for all electrodes in the recording.
Returns a matrix of [2*resolution-1,dim,dim] so that the correlations over
different time offsets may be inspected. Symmetry of the method is used to
double the resolution of method in comparable runtime.
"""
function crossCorrelogram(recording::Recording, range::Float64, delta::Float64, resolution::Int; rejectSimultaneous::Bool=true, zeroDiag::Bool=true)
    channels = recording.channels
    times = recording.times
    config = recording.config
    dim = config["channels"]

    #remove the simultaneous spikes from the channels and times, if the option is true
    if rejectSimultaneous
        nonSimultaneousSpikes = findNotCoincident(times)
        channels = channels[nonSimultaneousSpikes]
        times = times[nonSimultaneousSpikes]
    end

    #println(channels)
    #println(times)

    #declare the array for values to go into
    corrs = zeros(resolution*2-1,dim,dim)

    #set up the array of offset values to do the correlation over
    timeOffsets = linspace(0,range,resolution)

    #step through the different time periods which we are examining for correlations
    for offsetIndex = 1:resolution
      #retrieve the appropriate offset time for the correlation to use
      timeOffset = timeOffsets[offsetIndex]

      #define the temporal period where correlations will be looked for
      startTime = timeOffset - delta
      stopTime = timeOffset + delta

      #println("Time start: ", startTime*1000, " (ms), time stop: ", stopTime*1000, " (ms)")
        #do the cross-correlation outer sum
        for i = 2:length(channels)-1
            currentTime = times[i]

            #now find the indices where this window may occur
            startIndex = i
            #println("spike index ", i)

            #look for the starting index of the time window
            #or determine if it doesn't exist
            #see if we need to move backwards or forwards
            if times[i] - currentTime > startTime
                #if it's true, then we need to move to the past
                #decrement this index  until we exceed the start time
                while times[startIndex] - currentTime >= startTime
                    startIndex -= 1
                    if startIndex == 0
                      break
                    end
                end

                startIndex += 1

                #if we are not within the time window now, it doesn't exist
                if times[startIndex] - currentTime > stopTime
                    #println("No time window -, SI ", startIndex)
                    continue
                end
                #else, we have found the starting index in the window

            #do the same thing, but moving forward in time
            elseif times[i] - currentTime <= startTime
                #increment this index  until the start of the time window is exceeded.
                while startIndex < length(times) && times[startIndex] - currentTime < startTime
                    startIndex += 1
                end

                #if we are also past the stop time, or at the EOF, the time window doesn't exist
                if startIndex >= length(times) || times[startIndex] - currentTime > stopTime
                    #println("No time window +")
                    continue
                end
            end
            #println("StartIndex ", startIndex)

            #now that the start of the time window has been found, or the window has been rejected
            #find the ending index of the time window (only moves forward)
            stopIndex = startIndex
            while stopIndex < length(times) && times[stopIndex] - currentTime <= stopTime
                stopIndex += 1
            end

            #now, the index will either be one past the stop window, or at the EOF
            #if it is one past, move it back to the proper position
            if times[stopIndex] - currentTime > stopTime
                stopIndex -= 1
            end

            #these two indices determine the portion of the matrix which contains a correlation

            #println("StopIndex ", stopIndex)

            #do the cross-correlation inner sum
            for j = startIndex:stopIndex
                reference = channels[i]
                source = channels[j]
                #make sure we aren't correlating a signal to itself
                if i==j
                    continue
                end
                corrs[offsetIndex+(resolution-1),reference, source] += 1
            end

            #print("\n")
        end
    end

    #set the self-correlating diagonal elements to zero
    if zeroDiag
        corrs = zeroDiagonal!(corrs)
    end

    #because of the symmetry of cross-correlation, we can flip the matrix to return the CC for negative times.
    for timeIndex = 1:resolution-1
        corrs[timeIndex,:,:] = transpose(corrs[size(corrs,1)-timeIndex,:,:])
    end

    return corrs

end

"""
Normalize an adjacency matrix, by mutiplying each element with the inverse square root
of the sum of counts for both channels being correlated (see formula).
If the count is zero, the normalization multiplies by zero.
"""
function normalizeCounts(pathways::Array, recording::Recording)

    spikesByChannel = counts(recording.channels, recording.config["channels"])

    products = sqrt.(spikesByChannel * spikesByChannel').^-1
    #remove infinites created by dead electrodes
    products[isinf.(products)] = 0

  return pathways.*products
end


"""
Change the diagonal elements set to zero in a stack of 2D matrices.
"""
function zeroDiagonal!(matrix::Array{Float64,3})
    for i = 1:size(matrix,1)
        inds = diagind(matrix[i,:,:])
        matrix[inds] = 0
    end
    return matrix
end

"""
Change a matrix's diagonal elements set to zero.
"""
function zeroDiagonal!(matrix::Array{Float64,2})
    inds = diagind(matrix)
    matrix[inds] = 0
    return matrix
end

"""
Carry out the cross-correlation for all electrodes in the recording.
Automatically attempts to calculate the correct number of sampling points, to evenly divide the sampling period with no gaps.
Can override this and other behaviors with optional keyword arguments.
Return only the maximum correlation values from the time period.
"""
function crossCorrelate(recording::Recording, range::Float64, delta::Float64; resolution::Int=-1, normalized::Bool=true, rejectSimultaneous::Bool=true, zeroDiag::Bool=true)

    #if resolution is left at the default value, then automatically calculate the right number of points
    if resolution == -1
        resolution = convert(Int,round(range/delta))
    end

    ccMat = maximum(crossCorrelogram(recording,range,delta,resolution,rejectSimultaneous=rejectSimultaneous),1)[1,:,:]
    if normalized
        ccMat = normalizeCounts(ccMat, recording)
    end

    if zeroDiag
        ccMat = zeroDiagonal!(ccMat)
    end

    return ccMat

end

"""
Determine the mean and variance of a distribution of surrogate data, created by shuffling (see ShuffleRecording).
Returns a tuple of (means, std. devs) matrices, for each element in the adjacency matrix.
If the full distribution is desired, set returnFull to true.
"""
function surrogateDistribution(analysisFunc::Function, shuffleFunc::Function, recording::Recording; numSurrogates::Int=100, returnFull::Bool=false, textOutput::Bool=false, parallelize::Bool=true)
    config = recording.config
    dim = config["channels"]

    results = zeros(numSurrogates, dim, dim)
    if textOutput println("Computing surrogates (", numSurrogates," total) -") end

    #julia's pmap function
    parallelize? np = nprocs() : np = 1 # determine the number of processes available
    n = numSurrogates

    func = x->analysisFunc(shuffleFunc(x))
    dispatch = Array{Future,1}(n)

    for i in 1:n
        dispatch[i] = remotecall(func,mod1(i,np),recording)
    end

    wait(dispatch[end])

    for i in 1:n
        results[i,:,:] = fetch(dispatch[i])
    end

    if textOutput print("\n") end

    if returnFull
      return results
    end

    mus = squeeze(mean(results,1),1)
    stddevs = squeeze(std(results,1),1)
    return (mus, stddevs)
end

"""
Carry out the cross-correlation and surrogate tests on a recording.
Return a matrix representing experimental, mean, and std deviation compared to surrogate data.
"""
function full_CC(file::Recording, shuffleFunc::Function, range::Real, delta::Real, numSurrogates::Int=100)

    dim = file.config["channels"]
    results = zeros(3,dim,dim)
    func = x->crossCorrelate(x,range,delta)

    results[1,:,:] = func(file)
    surrogates = surrogateDistribution(func, shuffleFunc, file)
    results[2,:,:] = surrogates[1]
    results[3,:,:] = surrogates[2]

    return results
end
