#define an abstract type which the recordings will be subtypes of
abstract type Recording end

"""
    type MEARecording(channels, times, config)

Contains the data for an MEA recording: the list of spike times (times) and the channel they occurred on (channels).
Indicates whether or not the recording is stimulated, and if it is, the list of stimulus times and the list indices immediately after they occurred.
Config contains information on the recording. Requires information on the number of channels, and active electrodes.
"""
type MEARecording <: Recording
    #A type which carries the data from a recording and information which is needed to process it.
    channels::Array{Int,1}
    times::Array{Float64,1}

    #relevant information on the culture itself
    config::Dict{String,Any}
end

"""
    type StimMEARecording(channels, times, stimTimes, stimIndices, stimElectrodes, config)

Contains the data for an MEA recording: the list of spike times (times) and the channel they occurred on (channels).
Indicates whether or not the recording is stimulated, and if it is, the list of stimulus times and the list indices immediately after they occurred.
Config contains information on the recording. Requires information on the number of channels, and active electrodes.
"""
type StimMEARecording <: Recording
    #A type which carries the data from a recording and information which is needed to process it.
    channels::Array{Int,1}
    times::Array{Float64,1}

    #stimulus information
    #the times at which stimuli occurred
    stimTimes::Array{Float64,1}
    #the channel/time index which comes immediately after the stimulus
    stimIndices
    #the electrode which was stimulated
    stimChannels

    #relevant information on the culture itself
    config::Dict{String,Any}
end

function convert(t::Type{MEARecording}, r::StimMEARecording)
    return MEARecording(r.channels, r.times, r.config)
end

"""
A structure for efficiently storing joint distributions of arbitrarily-long state vectors and future values.
"""
mutable struct JointDist
    dist::Array{SparseMatrixCSC{Int64, Int64},1}
    delay::Int
    offset::Int
    expansion_i::Array{Int,1}
    expansion_j::Array{Int,1}
    size_i::Int
    size_j::Int

    JointDist(delay::Int, offset::Int, states::Int, expansion_i::Array{Int,1},  expansion_j::Array{Int,1}, size_i::Int, size_j::Int) =
        if length(expansion_i) != delay || length(expansion_j) != delay
            error("Expansion must match delay embedding.")
        else
            jd = new(Array{SparseMatrixCSC{Int64, Int64},1}(states), delay, offset, expansion_i, expansion_j, size_i, size_j)
            for i in 1:states
                jd.dist[i] = spzeros(size_i,size_j)
            end
            return jd
        end
end

"""
Internal function for incrementing the correct location inside a joint distribution given an event and 2 state vectors.
"""
function embed!(joint_dist::JointDist, state::Int, eventi::Array{Int,1}, eventj::Array{Int,1})
    dist = joint_dist.dist

    (emi, emj) = encode(joint_dist, eventi, eventj)
    #increase the count of that embedded event
    dist[state+1][emi,emj] += 1
end

"""
Internal function for finding the correct location in a JD given 2 state vectors.
"""
function encode(joint_dist::JointDist, eventi::Array{Int,1}, eventj::Array{Int,1})
    expansion_i = joint_dist.expansion_i
    expansion_j = joint_dist.expansion_j
    delay = joint_dist.delay

    emi = 1
    emj = 1
    for k in 1:delay
        #use the bases based in a number system that uses the maximal counts on each channel to embed this event in a 2-D distribution
        emi += expansion_i[delay+1-k]*eventi[k]
        emj += expansion_j[delay+1-k]*eventj[k]
    end

    return (emi, emj)
end

"""
Internal function for calculating the state vectors corresponding to a location within a JD.
"""
function decode(joint_dist::JointDist, ec_i::Int, ec_j::Int)
    if ec_i > joint_dist.size_i || ec_j > joint_dist.size_i
        error("Event code can't correspond to this joint distribution.")
    end

    expansion_i = joint_dist.expansion_i
    expansion_j = joint_dist.expansion_j
    delay = joint_dist.delay

    event_i = zeros(Int, delay)
    event_j = zeros(Int, delay)
    ec_i -= 1
    ec_j -= 1
    for k in delay:-1:1
        event_i[delay+1-k] = div(ec_i, expansion_i[k])
        ec_i = mod(ec_i, expansion_i[k])

        event_j[delay+1-k] = div(ec_j, expansion_j[k])
        ec_j = mod(ec_j, expansion_j[k])
    end

    return (event_i, event_j)
end

"""
The number of embeddings contained within a JD.
"""
function n_embeddings(joint_dist::JointDist)
    n = Int(0)
    dist = joint_dist.dist
    for i in 1:length(dist)
        n += sum(dist[i])
    end
    return n
end

"""
Returns true if a joint distribution represents a trivial process (all zeros).
"""
function is_trivial(joint_dist::JointDist)
    dist = joint_dist.dist
    if n_embeddings(joint_dist) == sum([dist[i][1,1] for i in 1:length(dist)])
        return true
    else
        return false
    end
end

"""
Convert a joint distribution of counts into a table of expectations for the next state given the current embedding
"""
function expectation_dist(jd::JointDist)
    joint_dist = jd.dist
    #what firing states exist for us to predict?
    n_states = length(joint_dist)
    states = 1:n_states
    #set up the total counts and marginal distributions
    total = 0
    margins = Array{SparseVector{Int64,Int64},1}(n_states)
    expectation = Array{SparseMatrixCSC{Float64, Int64}}(n_states)

    #construct the marginal distributions / total count
    for state in states
        substate = joint_dist[state]
        #the number of events this substate holds
        total += sum(substate)
        #the marginal distribution - probability of each state given only its own history
        margins[state] = sparsevec(sum(substate,2))

        #set up the expectation arrays
        s_size = size(substate)
        expectation[state] = spzeros(Float64,s_size[1],s_size[2])
    end

    #then iterate through all the states, calculate the probabilities for each event,
    #and use them to calculate the transfer entropy.

    m, n = size(joint_dist[1])
    checked = spzeros(Bool,m,n)

    #find the expectation value for the next state given a combination of embeddings
    for state in states

        substate = joint_dist[state]
        rows = rowvals(substate)
        counts = nonzeros(substate)

        #iterate through the columns of the matrix
        for j in 1:n
            #and pull out the non-zero values for that column
            for k in nzrange(substate, j)
                #find the value
                count = counts[k]
                #find what row it relates to
                i = rows[k]

                #if this is true, we've already constructed the expectation for this value
                if checked[i,j]
                    continue
                end

                total = Int(0)

                #how many times was this embedding combination counted for all states?
                for state in states
                    total += joint_dist[state][i,j]
                end

                checked[i,j] = true

                for state in states
                    expectation[state][i,j] = joint_dist[state][i,j] / total
                end

            end
        end
    end

    return expectation
end

function expectation(jd::JointDist, ex::Array{SparseMatrixCSC{Float64, Int64}}, event_x::Array{Int,1}, event_y::Array{Int,1})
    (enc_x, enc_y) = encode(jd, event_x, event_y)
    return expectation(ex, enc_x, enc_y)
end

function expectation(ex::Array{SparseMatrixCSC{Float64, Int64}}, i_x::Int, i_y::Int)
    l = length(ex)
    values = zeros(Float64,l)
    for i in 1:l
        values[i] = ex[i][i_x,i_y]
    end
    return values
end

"""
The number of spikes in a recording.
"""
function length(rec::Recording)
    return size(rec.channels,1)
end

"""
Slice a recording into pieces with equal number of spikes.
"""
function islice(recording::MEARecording, slices::Int)
    n = length(recording)
    spikes = floor(Int,n/slices)
    recordings = Array{MEARecording,1}(slices)

    times = recording.times
    channels = recording.channels
    config = recording.config

    for i in 1:slices
        start = (i-1)*spikes+1
        stop = (i)*spikes
        recording = MEARecording(channels[start:stop], times[start:stop], config)
        recordings[i] = recording
    end

    return recordings
end

"""
Slice a recording into pieces with equal temporal length.
"""
function tslice(recording::MEARecording, slices::Int)
    times = recording.times
    channels = recording.channels
    config = recording.config

    cutoffs = linspace(times[1], times[end], slices+1)
    recordings = Array{MEARecording,1}(slices)
    n = length(recording)

    slice = 1
    start = 1
    stop = 1
    while slice <= slices
        #find the index to stop at
        while times[stop] < cutoffs[slice+1] && stop < n
            stop += 1
        end
        #create a recording from this subsection
        recording = MEARecording(channels[start:stop], times[start:stop], config)
        recordings[slice] = recording
        slice += 1
        #move the starting point
        start = stop + 1
    end
    return recordings
end

"""
Chop a subsection from the start of a recording out.
"""
function chop_rec(recording::MEARecording, stop::Int)
    n = length(recording)
    #snap stop to within recording length
    stop = min(stop,n)

    times = recording.times
    channels = recording.channels
    config = recording.config

    recording = MEARecording(channels[1:stop], times[1:stop], config)

    return recording
end
