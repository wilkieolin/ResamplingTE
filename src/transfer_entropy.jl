"""
Take a recording and convert it into a sparse vector of spike counts which occur in binned regions.
"""
function sp_count_vector(rec::Recording, bin::Float64, binary::Bool=false)
    if bin < 0
        error("Bin size must be positive.")
    end

    #when does the recording begin and end?
    start_t = rec.times[1]
    end_t = rec.times[end]
    #how long does it last?
    rec_time = start_t - end_t
    channels = rec.config["channels"]

    #create an iterator for each time bin, making sure the final spike will be counted
    time_iter = (start_t):bin:(end_t+bin)
    n_bins = length(time_iter)

    #create the dictionary that maps each time bin to its spike counts/channel
    counts = Dict{Int64,SparseVector{Int64,Int64}}()
    #this array holds the accumulated numer of spikes per each time bin
    current_counts = zeros(channels)
    #this array tracks the maximum number of spikes found in a bin
    maxima = zeros(channels)

    i = 1
    #in each bin where spikes are to be counted
    for (it,cutoff) in enumerate(time_iter)
        #move the index forward through each spike in this bin
        while i <= length(rec.times) && rec.times[i] <= cutoff
            #what channel are we looking at?
            ch = rec.channels[i]
            #increment the spike counter
            current_counts[ch] += 1
            #increment our place
            i += 1
        end

        if sum(current_counts) > 0
            #assign the current counter value into its bin
            if binary
                #if we're using a binary embedding then convert every int > 0 to 1
                current_counts = (current_counts .> 0) .* 1.0
            end

            counts[it] = sparse(current_counts)

            #did we find a new maximum?
            for j in 1:channels
                if current_counts[j] > maxima[j]
                    #assign the new maximum into the array
                    maxima[j] = current_counts[j]
                end
            end
        end
        #reset the bin spike-counter to 0
        fill!(current_counts,0)
    end

    #add the maximum counts in the -1 bin
    counts[-1] = sparse(maxima)
    #add an empty vector at the end so we know the original size of the recording
    if !(haskey(counts,n_bins))
        counts[n_bins] = spzeros(channels)
    end
    #add a reference for zeros to point to, showing there were no counts
    counts[0] = spzeros(channels)

    return counts
end

"""
sp_joint_dist(counts1::Array{<:Int,1}, counts2::Array{<:Int,1}, delay::Int)

Given two vectors that represent the number/existence of spikes in a series of time bins,
calculate a joint distribution repesenting the combined embedding of their states over time.
Sparse representation is used to create this distribution as the space of all possible states is enormous.
"""
function sp_joint_dist(counts::Dict{<:Int,<:SparseVector{<:Int,<:Int}}, delay::Int)
    return sp_joint_dist(counts,delay,0)
end

function sp_joint_dist(counts::Dict{<:Int,<:SparseVector{<:Int,<:Int}}, delay::Int, offset::Int)
    #what were the maximum bin counts per channel?
    maxima = full(counts[-1])
    #the length of the maxima is the number of channels in the recording
    channels = length(maxima)
    #the list of active channels which need to have their embeddings updated for each event
    active = falses(channels)
    #set up a list to hold the distribution of joint spike embeddings for each channel
    joint_dist = Array{JointDist,2}(channels,channels)

    #create a sorted list of keys to the bin-count dictionary we'll use to go through it
    #discard the -1 & 0 key since we just use those to pass information
    nz_bins = sort!(collect(keys(counts)))[3:end]

    #set up the array we'll use to keep track of which delay embedding will go where in the SP vector
    bases = maxima .+ 1
    #the outer sizes of the 1-D spvectors we embed the marginal distribution inside, based on maximal counts
    outersizes = bases .^ delay
    #and the bases we use to expand counts into this embedding
    base_expansion = zeros(Int64,channels,delay)
    for i in 1:channels, j in 1:delay
        base_expansion[i,j] = bases[i]^(j-1)
    end

    #set up the array which holds the joint distribution for each adjacency
    for i in 1:channels, j in 1:channels
        if i == j
            continue
        end
        joint_dist[i,j] = JointDist(delay, offset, bases[i], base_expansion[i,:], base_expansion[j,:], outersizes[i], outersizes[j])
    end

    #create the iterator we'll use to go through these events
    bin_it = start(nz_bins)
    bin = nz_bins[bin_it]

    #keep track of the keys which refer to the events which were recorded in the sparse counts
    history_l = delay+offset+1
    history = zeros(Int64,history_l)
    state = Int(0)
    embedding1 = zeros(Int,delay)
    embedding2 = zeros(Int,delay)
    #go through the sparse bins of spike counts over all channels

    #define a function we can use to embed multiple events into the marginal distribution
    circ_embed(x::Int) =
        for r in 1:x
            #increase the count of this event happening in the marginal distribution
            #println("Bins: ",history," l,",i)
            #embed!(joint_dist, counts, base_expansion, outersizes, maxima, history)
            #for each combination of channels in a recording
            for i in 1:channels, j in 1:channels
                #skip if there were no events recorded on this channel, we know trivially its TE will be 0
                if i == j || maxima[i] == 0 || (!active[i] && !active[j])
                    continue
                end
                event = false
                #if there's no events within the relevant window or at present then continue
                for k in 1:delay
                    if history[k] != 0
                        event = true
                    end
                end

                if !event && history[end] == 0
                    continue
                end

                #DB println("I: ",i," J: ",j)

                #what's the spiking state we're predicting?
                state = counts[history[end]][i]
                #what's the corresponding embedding?
                for k in 1:delay
                    embedding1[k] = counts[history[k]][i]
                    embedding2[k] = counts[history[k]][j]
                end
#                 print("Source: ")
#                 for k in 1:delay
#                     print(counts[history[k]][i]," ")
#                 end
#                 print("Dest: ")
#                 for k in 1:delay
#                     print(counts[history[k]][j]," ")
#                 end
#                 print("S: ",state," \n")

                embed!(joint_dist[i,j], state, embedding1, embedding2)

            end
            #move the keys along in time for the next embedding
            for i in 1:history_l-1
                history[i] = history[i+1]
            end
            #either the next event will be no spike, or this 0 will be replaced by n_bin
            history[end] = 0
        end


    while true
        #what's the next bin after this?
        (n_bin, n_bin_it) = next(nz_bins,bin_it)
        #DB println("Current bin: ",bin)

        #what's the current event we're looking at?
        history[end] = bin
        #DB println("BIN ",bin)
        #find if our embedding window had any recorded events
        for i in 1:history_l
            #the index of the possible event we're looking for
            event_ind = max(bin-(delay+offset)+(i-1),0)
            #did we record any events at this previous step?
            if haskey(counts,event_ind)
                #if so, add a reference to it
                history[i] = event_ind
                #find which channels in this event were active
                for j in counts[event_ind].nzind
                    active[j] = true
                end
            else
                #if not, refer to the 0 event which is no counts on any channel
                history[i] = 0
            end
        end

        #how much do we need to embed before running into the next event?
        l = min(delay+1,n_bin-bin)
        #now for each key in the window we found, use it to find the embedding for each joint distribution
        circ_embed(l)

        #move to the next bin in sequence
        #DB println("Next bin: ",n_bin)
        bin = n_bin
        bin_it = n_bin_it
        if done(nz_bins,bin_it)
            break
        end
        #reset the active channels marker
        fill!(active,false)
    end
    #once we are on the last event, run it out until it's embedded
    circ_embed(delay+1)
    #DB println("Done")

    #the number of embeddings we've skipped over now are all 0s
    embedding_sum = (nz_bins[end]-nz_bins[1])+delay+1
    #for every channel
    for i in 1:channels, j in 1:channels
        if i == j continue end

        dist = joint_dist[i,j].dist
        #sum the number of embeddings found for that combination
        c = 0
        for k in keys(dist)
            c += sum(dist[k])
        end
        #add the missing embeddings in the 0-0,0... place
        #DB println("Sum ",c)
        #DB println("ESUM ",embedding_sum)
        #DB println("Missing ",embedding_sum-c)
        dist[1][1,1] += (embedding_sum - c)
    end

    return joint_dist
end

"""
Calculates the marginal distribution of embeddings in each channel, appropriate for creating surrogate data.
"""
function reverse_marginal_dist(counts::Dict{<:Int,<:SparseVector{<:Int,<:Int}}, delay::Int)
    #what were the maximum bin counts per channel?
    maxima = full(counts[-1])
    #the length of the maxima is the number of channels in the recording
    channels = length(maxima)
    #the list of active channels which need to have their embeddings updated for each event
    active = falses(channels)
    #set up a list to hold the distribution of spike embeddings for each channel
    marginal_dist = Array{Dict{Array{Int,1},Array{Int,1}},1}(channels)

    #create a sorted list of keys to the bin-count dictionary we'll use to go through it
    #discard the -1 & 0 key since we just use those to pass information
    nz_bins = sort!(collect(keys(counts)))[3:end]

    #set up the list which holds the marginal distributions for each channel
    for i in 1:channels
        marginal_dist[i] = Dict{Array{Int,1},Array{Int,1}}()
    end

    #create the iterator we'll use to go through these events
    bin_it = start(nz_bins)
    bin = nz_bins[bin_it]

    #keep track of the keys which refer to the events which were recorded in the sparse counts
    history_l = delay+1
    history = zeros(Int64,history_l)

    #go through the sparse bins of spike counts over all channels

    #define a helper function we can use to embed multiple events into the marginal distribution
    circ_embed(x::Int) =
        for r in 1:x
            #increase the count of this event happening in the marginal distribution
            #DB println("Bins: ",history," l,",i)
            #embed!(marginal_dist, counts, base_expansion, outersizes, maxima, history)
            for i in 1:channels
                #the local variable to store the embedding for each event being counted
                embedding = zeros(Int64,delay)

                if maxima[i] == 0 || !active[i]
                    continue
                end
                this_channel = marginal_dist[i]

                #what's the spiking state leading up to the current event?
                for j in 1:delay
                    embedding[j] = counts[history[j]][i]
                end
                state = counts[history[end]][i]
                #DB println("E: ",embedding, "S: ",state)

                #does this state have a marginal distribution already in the dictionary?
                if !haskey(this_channel, embedding)
                    #DB println("Creating ", embedding)
                    this_channel[embedding] = zeros(Int,maxima[i]+1)
                    #DB println(collect(keys(this_channel)))
                end

                #increase the count of that event
                this_channel[embedding][state + 1] += 1
                #DB print("\n")
            end
            #move the keys along in time for the next embedding
            history = circshift(history,-1)
            #either the next event will be no spike, or this 0 will be replaced by n_bin
            history[end] = 0
        end

    while true
        #what's the next bin after this?
        (n_bin, n_bin_it) = next(nz_bins,bin_it)
        #DB println("Current bin: ",bin)

        #what's the current event we're looking at?
        history[end] = bin
        #find if our embedding window had any recorded events
        for i in 1:history_l
            #the index of the possible event we're looking for
            event_ind = max(bin-(delay-(i-1)),0)
            #did we record any events at this previous step?
            if haskey(counts,event_ind)
                #if so, add a reference to it
                history[i] = event_ind
                #find which channels in this event were active
                for j in counts[event_ind].nzind
                    active[j] = true
                end
            else
                #if not, refer to the 0 event which is no counts on any channel
                history[i] = 0
            end
        end

        #how much do we need to embed before running into the next event?
        l = min(history_l,n_bin-bin)
        #now for each key in the window we found, use it to find the embedding for each joint distribution
        circ_embed(l)

        #move to the next bin in sequence
        #DB println("Next bin: ",n_bin)
        bin = n_bin
        bin_it = n_bin_it
        if done(nz_bins,bin_it)
            break
        end
        #reset the active channels marker
        fill!(active,false)
    end

    #once we are on the last event, run it out until it's embedded
    circ_embed(history_l)
    #DB println("Done")

    #the number of embeddings we've skipped over now are all 0s
    embedding_sum = (nz_bins[end]-nz_bins[1])+history_l

    no_event = zeros(Int,delay)
    #for every channel
    for i in 1:channels
        dist = marginal_dist[i]
        #sum the number of embeddings found for that combination
        c = 0
        for k in keys(dist)
            c += sum(dist[k])
        end
        #add the missing embeddings in the 0-0,0... place
        if !haskey(dist,no_event)
            dist[no_event] = spzeros(1)
        end
        #DB println("Sum ",c)
        #DB println("ESUM ",embedding_sum)
        #DB println("Missing ",embedding_sum-c)
        dist[no_event][1] += (embedding_sum - c)
    end

    return marginal_dist

end

"""
Analyzes a sparse count vector, returns the probability of spiking on each channel.
"""
function simple_dist(spv::Dict{Int,SparseVector{Int,Int}})
    n = length(spv[0])
    max = spv[-1]
    counts = Dict{Int,Array{Float64,1}}()
    #prob = Dict{Int,Array{Int,1}}()
    spkeys = sort!(collect(keys(spv)))[3:end]

    #set up the counts for each channel, sizes corresponding to the number of possible events
    for ch in 1:n
        counts[ch] = zeros(max[ch])
    end

    #for each event recorded
    for key in spkeys
        #for each active channel
        for ch in spv[key].nzind
            #find the state recorded
            val = spv[key][ch]
            #increment the record of this state occurring
            counts[ch][val] += 1
        end
    end

    for ch in 1:n
        dist = counts[ch]
        #the total number of events recorded on this channel
        total = sum(dist)
        #calculate the probabilty of each occurring
        if total > 0
            dist ./= total
        end
    end

    return counts
end

"""
Randomly chooses an outcome from a pdf, with the chance an outcome is selected proportional to its density value
"""
function SUS(expectations::Array{<:Real,1})
    p = rand()
    psum = Float64(0)
    j = Int(0)
    n = length(expectations)

    while psum < p && j < n
        j += 1
        psum += expectations[j]
    end

    #return a '0' state if the expectations are invalid / don't sum to 1
    if psum < p
        return 1
    else
        return j
    end
end

"""
Creates a resample of a joint distribution, by following the Markov Chain which the dynamics of the signal creates.
"""
function joint_dist_resample(x::JointDist, y::JointDist, offsets::Array{Int,1}, proportion::Real=1.0)
    delay = x.delay
    if delay != y.delay
        error("Delays must be equal.")
    end
    if x.offset != 0 || y.offset != 0
        error("To resample from a distribution, its offset must be 0.")
    end

    samples = n_embeddings(x) * proportion
    n_states_x = length(x.dist)
    n_states_y = length(y.dist)
    n_offsets = length(offsets)
    max_offset = maximum(offsets)
    l = delay + max_offset

    #create an array that will hold the current state of the signal being resampled
    #start everything off at zero - the usual non-firing state
    signal_x = zeros(Int,l)
    signal_y = zeros(Int,l)
    embedding_x = zeros(Int,delay)
    embedding_y = zeros(Int,delay)
    next_x = Int(0)
    next_y = Int(0)
    proposition_x = zeros(Int,delay)
    proposition_y = zeros(Int,delay)

    #create the new joint distributions
    jdx = Array{JointDist,1}(n_offsets)
    jdy = Array{JointDist,1}(n_offsets)
    for (i,offset) in enumerate(offsets)
        jdx[i] = JointDist(delay, offset, n_states_x, x.expansion_i, x.expansion_j, x.size_i, x.size_j)
        jdy[i] = JointDist(delay, offset, n_states_y, y.expansion_i, y.expansion_j, y.size_i, y.size_j)
    end

    ex = expectation_dist(x)
    ey = expectation_dist(y)

    s = Int(0)
    i_x = Int(0)
    i_y = Int(0)
    r = Int(0)

    while s < samples
        #embed the current states at the offsets we're inspecting
        for (i,offset) in enumerate(offsets)
            #cut down the signal to the embedding
            for j in 1:delay
                embedding_x[j] = signal_x[offset+j]
                embedding_y[j] = signal_y[offset+j]
            end
            #embed this joint sample in the corresponding JD
            embed!(jdx[i], next_x, embedding_x, embedding_y)
            embed!(jdy[i], next_y, embedding_y, embedding_x)
        end

        #move forward in time
        for i in 1:l-1
            signal_x[i] = signal_x[i+1]
            signal_y[i] = signal_y[i+1]
        end
        signal_x[l] = next_x
        signal_y[l] = next_y

        #given the new joint state, find out what values we can expect next
        for i in 1:delay
            embedding_x[i] = signal_x[max_offset+i]
            embedding_y[i] = signal_y[max_offset+i]
        end
        (i_x, i_y) = encode(x, embedding_x, embedding_y)

        ex_x = expectation(ex, i_x, i_y)
        ex_y = expectation(ey, i_y, i_x)

        #if there are no values we've somehow wandered into an undefined state, possibly the joint distribution is bad
        if sum(ex_x) == 0.0 && sum(ex_y) == 0
            println("SX: ", signal_x," | SY: ",signal_y)
            error("Unknown state encountered. Check joint distribution / original signal.")
        end

        #now, attempt to find a new valid state
        c = Int(0)

        for i in 1:delay-1
            proposition_x[i] = embedding_x[i+1]
            proposition_y[i] = embedding_y[i+1]
        end

        while true
            #randomly choose a new X and Y state from the possible states
            #keep in mind spikes are 0-indexed and the expectation array is 1-indexed, subtract 1
            proposition_x[delay] = SUS(ex_x) - 1
            proposition_y[delay] = SUS(ex_y) - 1

            #check and see if this would be a valid joint state
            (i_x, i_y) = encode(x, proposition_x, proposition_y)
            ex_px = expectation(ex, i_x, i_y)
            ex_py = expectation(ey, i_y, i_x)

            #if there's been a valid recorded state in this configuration, it will have a non-zero expectation value
            if sum(ex_px) != 0.0 && sum(ex_py) != 0.0
                break
            end

            c += 1
            if c > 100000
                #if we can't find a valid joint state for some reason, reset the signal to 0
                proposition_x[delay] = 0
                proposition_y[delay] = 0
                r += 1
                break
            end
        end

        #accept the new state we've found
        next_x = proposition_x[delay]
        next_y = proposition_y[delay]
        s += 1
    end

    if r > 0
        println("Had to reset ",r," times.")
    end

    return (jdx, jdy)
end

"""
Marginal distribution surrogates: uses the marginal embedding distributions of each channel to synthesize surrogate recordings.
Returns sparse vector for use with entropy methods.
"""
function MDS(recording::MEARecording, bin::Real, delay::Int, relsize::Real=1.0)
    scv = sp_count_vector(recording,bin)
    n = length(recording)
    md = reverse_marginal_dist(scv,delay)
    return MDS(md, round(Int,relsize*n))
end

function MDS(marginal_dists::Array{Dict{Array{Int,1},Array{Int,1}},1}, spikes_limit::Int)
    channels = size(marginal_dists,1)
    #figure out the delay size
    delay = size(first(keys(marginal_dists[1])),1)
    #println("Delay, ",delay)

    #set up an array to hold the probabilities of the next event given the past state for each channel
    state_expectations = Array{Dict{Array{Int,1},Array{Float64,1}}}(channels)
    [state_expectations[ch] = Dict{Array{Int,1},Array{Float64,1}}() for ch in 1:channels]

    #find the chance of each state for every channel
    for ch in 1:channels
        dist = marginal_dists[ch]
        #find the counts of each state occurring
        for em in keys(dist)
            total = sum(dist[em])
            state_expectations[ch][em] = dist[em] ./ total
        end
    end

    #begin constructing a new sparse vector encoding a generated recording
    sp_vec = Dict{Int64,SparseVector{Int64,Int64}}()
    #start all channels off at the 0 state (quiet)
    states = zeros(Int64,channels,delay)
    state = zeros(Int64,delay)
    no_event = zeros(Int64,delay)
    maxima = zeros(Int64,channels)
    #find the index for the first generated event
    i = delay + 1
    #keep track of the number of spikes we've generated
    spikes = Int64(0)
    while spikes < spikes_limit
        #DB println(states)
        for ch in 1:channels
            #get the current state of the signal
            for j in 1:delay
                state[j] = states[ch,j]
            end
            #what are the choices for the next state given this embedding?
            choices = state_expectations[ch][state]

            #see if this is a dead channel, and skip it if so (only choice is 0,0,0...)
            if choices[1] == 1.0 && state == no_event
                continue
            end

            #use stochastic universal selection to choose between possible embedding events for this channel and current state
            #number of spikes are one-indexed, subtract down to 0 index
            next_state = SUS(choices) - 1

            #DB println("C: ",state," ",next_state)

            #if we've selected a non-zero event, add it to the generated spike train
            if next_state > 0
                #increase the number of spikes we've generated
                spikes += next_state
                #place a spike in the sparse vector for that channel
                #create the dictionary for this timestep if we need to
                if !haskey(sp_vec,i)
                    sp_vec[i] = spzeros(channels)
                end
                sp_vec[i][ch] = next_state

                #update the sp vec's maxima for TE
                if next_state > maxima[ch]
                    maxima[ch] = next_state
                end
            end
            #update the embedding to the new state
            #shift the state backwards
            for j in 1:delay-1
                states[ch,j] = states[ch,j+1]
            end
            #change the end to the new value
            states[ch,delay] = next_state

        end
        i += 1
        #DB println(states)
    end

    #add the reference information
    sp_vec[0] = spzeros(channels)
    sp_vec[-1] = maxima

    return sp_vec
end

"""
Given a matrix or single joint distribution, calculate the transfer entropy.
"""
function te_calc(joint_dists::Array{JointDist,2})
    n = size(joint_dists,2)
    te = zeros(Float64,n,n)
    #traveling along the upper triangle
    for i in 2:n
        for j in 1:(i-1)
            #calculate each pair of TE values
            #print(i," ",j," |")
            jdx = joint_dists[i,j]
            jdy = joint_dists[j,i]
            te[i,j] = te_calc(jdx)
            te[j,i] = te_calc(jdy)
        end
    end

    return te
end

function te_calc(jd::JointDist)

    joint_dist = jd.dist
    #what firing states exist for us to predict?
    n_states = length(joint_dist)
    states = 1:n_states
    #set up the total counts and marginal distributions
    total = 0
    margins = Array{SparseVector{Int64,Int64},1}(n_states)

    #construct the marginal distributions / total count
    for state in states
        substate = joint_dist[state]
        #the number of events this substate holds
        total += sum(substate)
        #the marginal distribution - probability of each state given only its own history
        margins[state] = sparsevec(sum(substate,2))
    end

    te = 0

    #then iterate through all the states, calculate the probabilities for each event,
    #and use them to calculate the transfer entropy.
    for state in states
        substate = joint_dist[state]
        m, n = size(substate)

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

                #what's the total joint probability of this state?
                joint_prob = count/total
                #what's the conditional probability of this state, given all other possible states
                cond_total = sum([joint_dist[state][i,j] for state in states])
                cond_prob = count/cond_total
                #what's the probability independent of the other neuron?
                ind_count = margins[state][i]
                ind_total = sum([margins[state][i] for state in states])
                ind_prob = ind_count/ind_total
                #println("i,j: ",i,j," || jp, cp, ip ",joint_prob," ",cond_prob," ",ind_prob)

                te += joint_prob*log(2,cond_prob/ind_prob)
            end
        end
    end

    return te
end

function sp_entropy(margins::Dict{Int64,SparseVector{UInt64,Int64}})
  entropy = Dict{Int64,Float64}()

  total = 0
  for state in keys(margins)
    for val in margins[state].nzval
      total += val
    end
  end

  for state in keys(margins)
    ent = 0
    for val in margins[state].nvzal
      ent += (val/total)
    end
    entropy[state] = ent
  end

  return entropy
end

"""
Given a recording with channels/times, encode it into a vector of counts / time bin,
and then calculate the transfer entropy for each combination of channels.
"""
function transfer_entropy(rec::Recording, bin_size::Real, delay::Int, offset::Int, binary::Bool=false)
    #how many channels in the recording?
    n = rec.config["channels"]
    #generate the counts / bin for each channel
    count_vecs = sp_count_vector(rec,bin_size,binary)
    joint_dists = sp_joint_dist(count_vecs,delay,offset)
    te = te_calc(joint_dists)

    return te
end

function transfer_entropy(count_vecs::Dict{Int,SparseVector{Int,Int}}, bin_size::Real, delay::Int, offset::Int, binary::Bool=false)
    #how many channels in the recording?
    chs = length(count_vecs[0])
    #generate the counts / bin for each channel

    joint_dists = sp_joint_dist(count_vecs,delay,offset)
    te = te_calc(joint_dists)

    return te
end

function transfer_entropy(rec::Recording, bin_size::Real, delay::Int, offsets::Union{Array{<:Int,1},UnitRange{<:Int}}; binary::Bool=false, delays::Bool=false, full_delays::Bool=false)
    ws = workers()
    np = length(ws)

    n = length(offsets)
    dim = rec.config["channels"]
    futures = Array{Future,1}(n)
    results = Array{Float64,3}(n,dim,dim)
    for (i,offset) in enumerate(offsets)
        w = ws[mod1(i,np)]
        futures[i] = remotecall(transfer_entropy, w, rec, bin_size, delay, offset)
    end
    wait(futures[end])

    for i in 1:n
      results[i,:,:] = fetch(futures[i])
    end

    te_peaks = zeros(Float64,dim,dim)

    #full_delays overrides the more limited argument
    delays = delays && !full_delays
    if delays
        te_delays = zeros(Int64,dim,dim)
    end

    for i in 1:dim, j in 1:dim
        d = indmax(results[:,i,j])
        te_peaks[i,j] = results[d,i,j]
        if delays
            te_delays[i,j] = d - 1
        end
    end

    if full_delays
        return (te_peaks, results)
    elseif delays
        return (te_peaks, te_delays)
    else
        return te_peaks
    end
end

"""
Given a collection of experimental joint distributions for each channel-pair,
resample these values using the MCMC method and return the resampled TE values.
"""
function resample_TE(jds::Array{JointDist,2}, all_offsets::Array{Int,2}, proportion::Real=1.0)
    n = size(jds,2)
    rtes = zeros(Float64,n,n)
    offsets = zeros(Int,2)
    #print("R: ")
    for i in 2:n
        for j in 1:(i-1)
            if (is_trivial(jds[i,j]) && is_trivial(jds[i,j]))
                continue
            end

            #find the number of offfsets which we need to calculate over
            offsets[1] = all_offsets[i,j]
            offsets[2] = all_offsets[j,i]
            uoffsets = unique(offsets)
            (rjdx, rjdy) = joint_dist_resample(jds[i,j],jds[j,i],uoffsets,proportion)

            if length(uoffsets) == 1
                rtes[i,j] = te_calc(rjdx[1])
                rtes[j,i] = te_calc(rjdy[1])
            elseif length(uoffsets) == 2
                #check we've got the right offsets, they should be because unique preserves order
                if rjdx[1].offset == offsets[1] && rjdy[2].offset == offsets[2]
                    rtes[i,j] = te_calc(rjdx[1])
                    rtes[j,i] = te_calc(rjdy[2])
                else
                    println("Offsets out of order / miscalculated.")
                end
            end
        end
        #print(i," ")
    end

    return rtes
end

"""
The full resampling transfer entropy method, which take a recording, finds its experimental TE values,
resamples those values, and returns a 3-dimensional matrix. The first dimension corresponds to the sample of TE values,
the second and third to the dimensions of the adjacency matrix.

Parallelizes to the number of threads available.
"""
function resampling_TE(rec::Recording, ns::Int, bin_size::Real, delay::Int, offsets::Union{Array{<:Int,1},UnitRange{<:Int}}; proportion::Real=1.0, binary::Bool=false, return_offsets::Bool=false, full_offsets::Bool=false)
    ws = workers()
    np = length(ws)

    n = length(offsets)
    dim = rec.config["channels"]
    results = zeros(Float64,ns+1,dim,dim)

    futures = Array{Future,1}(n)
    #array to hold the joint distributions for every offset
    joint_dists = Array{Array{JointDist,2},1}(n)
    #the peak experimental value of transfer entropy over offsets
    peaks = zeros(Float64,dim,dim)
    #its corresponding delay index
    peak_offsets = ones(Int64,dim,dim)

    rs_futures = Array{Future,1}(ns)

    #first, generate the experimental TE values
    te = zeros(Float64,n,dim,dim)
    count_vecs = sp_count_vector(rec,bin_size,binary)

    #calculate each offset's joint distribution and resulting TE values
    for (i,offset) in enumerate(offsets)
        w = ws[mod1(i,np)]
        futures[i] = remotecall(sp_joint_dist,w,count_vecs,delay,offset)
    end

    wait(futures[end])
    for i in 1:n
        joint_dists[i] = fetch(futures[i])
    end

    #calculate each offset's TE values and search for the peak
    for i in 1:length(offsets)
        #for each possible non-self connection in the adjacency matrix
        for j in 2:dim
            for k in 1:(j-1)
                #print(i," ",j," |")
                #find the directed information flow for upper (u) and lower (l) joint dists
                jdu = joint_dists[i][j,k]
                jdl = joint_dists[i][k,j]
                teu = te_calc(jdu)
                tel = te_calc(jdl)

                te[i,j,k] = teu
                te[i,k,j] = tel

                #update the peak values and their corresponding offsets
                if teu > peaks[j,k]
                    peaks[j,k] = teu
                    peak_offsets[j,k] = i
                end

                if tel > peaks[k,j]
                    peaks[k,j] = tel
                    peak_offsets[k,j] = i
                end
            end
        end
    end

    #tstart = time()
    #now resample the experimental conditions, as captured by the JDs
    for i in 1:ns
        w = ws[mod1(i,np)]
        rs_futures[i] = remotecall(resample_TE, w, joint_dists[1], peak_offsets, proportion)
    end

    wait(rs_futures[end])

    #tend = time()
    #println("Took ",tend-tstart)

    #collect all the resampled results
    for i in 1:ns
        results[i+1,:,:] = fetch(rs_futures[i])
    end
    #collect the original result
    for i in 1:dim, j in 1:dim
        if i == j
            continue
        end
        results[1,:,:] = peaks
    end

    if full_offsets
        return (results, te)
    elseif return_offsets
        return (results, peak_offsets)
    else
        return results
    end
end

"""
Using the marginal distribution surrogate (MDS) method, calculate TE values which
represent the null hypothesis (no connectivity) for a recording.
"""
function surr_MDS(recording::Recording, ns::Int, bin::Real, delay::Int, offsets::Union{UnitRange{<:Int},Array{<:Int,1}}, embeddings::Real=1.0)
    #checkworkers()

    dim = recording.config["channels"]
    no = length(offsets)
    ws = workers()
    np = length(ws)
    futures = Array{Future,1}(ns)
    surrogates = zeros(ns,dim,dim)

    for (i,s) in enumerate(1:ns)
        w = ws[mod1(i,np)]
        offset = offsets[mod1(i,no)]
        func = x->transfer_entropy(MDS(x,bin,delay,embeddings), bin, delay, offset)
        futures[i] = remotecall(func,w,recording)
    end

    wait(futures[end])
    for i in 1:ns
        surrogates[i,:,:] = fetch(futures[i])
    end

    return surrogates
end

function checkworkers()
    np = nprocs()
    minworkers = 16
    if np == 1
        warn("Not running in parallel - analysis will be very slow.")
    if np < minworkers
        warn(string("Running in parallel, but at least ", minworkers, " workers are recommended."))
    end
end

function surr_TE(file::Recording, shuffleFunc::Function, time::Real, delay::Int, offset::Int=0, binary::Bool=false)
    surr_TE(file, shuffleFunc, time, delay, [offset], binary)
end

function surr_TE(file::Recording, ns::Int, shuffleFunc::Function, time::Real, delay::Int, offsets::Union{UnitRange{<:Int},Array{<:Int,1}}, binary::Bool=false)
    #checkworkers()

    no = length(offsets)
    ws = workers()
    np = length(ws)

    if typeof(offsets) <: UnitRange{<:Int}
        offsets = collect(offsets)
    end

    dim = file.config["channels"]
    futures = Array{Future,1}(ns)
    surrogates = zeros(ns,dim,dim)

    for i in 1:ns
        w = ws[mod1(i,np)]
        offset = offsets[mod1(i,no)]
        func = x->transfer_entropy(shuffleFunc(x), time, delay, offset, binary)
        futures[i] = remotecall(func,w,file)
    end

    wait(futures[end])
    for i in 1:ns
        surrogates[i,:,:] = fetch(futures[i])
    end

    return surrogates
end

#a dummy function to pass indicating we'd like to use the MDS methods
function MCMC()
    return
end

function MCMC_TE(file::Recording, time::Real, delay::Int, offsets::Union{Array{Int,1},UnitRange{Int}}; surrogates::Int=32)
    exp = resampling_TE(file, surrogates, time, delay, offsets)
    surr = surr_MDS(file, surrogates, time, delay, offsets)

    return (exp, surr)
end

function full_TE(file::Recording, shuffleFunc::Function, time::Real, delay::Int, offset::Int, binary::Bool=false)
    full_TE(file, shuffleFunc, time, delay, [offset], binary)
end

function full_TE(rec::MEARecording, shuffleFunc::Function, bin::Real, delay::Int, offsets::Union{Array{Int,1},UnitRange{Int}}; peak::Bool=false, delays::Bool=false, full_delays::Bool=false, surrogates::Int=32*2, full_surrogates::Bool=false, binary::Bool=false)
    dim = rec.config["channels"]
    results = Array{Float64}(3,dim,dim)
    return_d = full_delays || delays

    #create a single surrogate distribution, as results will remain unchanged by offsets
    if shuffleFunc == MCMC
        surrs = surr_MDS(rec, surrogates, bin, delay, offsets)
    else
        surrs = surr_TE(rec, surrogates, shuffleFunc, bin, delay, offsets)
    end

    #take either the peak values or mean values as the surrogate threshold
    if peak
        threshold = squeeze(maximum(surrs,1),1)
    else
        threshold = squeeze(mean(surrs,1),1)
    end
    devs = squeeze(std(surrs,1),1)

    te = transfer_entropy(rec, bin, delay, offsets, binary=binary, delays=delays, full_delays=full_delays)

    if return_d
      te_peaks = te[1]
    else
      te_peaks = te
    end

    results[1,:,:] = te_peaks
    results[2,:,:] = threshold
    results[3,:,:] = devs


    if return_d && full_surrogates
        return (results, te[2], surrs)
    elseif return_d
        return (results, te[2])
    else
        return results
    end
end

function te_differences(tev::Array{Array{Float64,2},2})
    n = size(tev,1)
    m = size(tev,2)

    diffs = zeros(Float64, n, m)

    for i in 1:n, j in 1:m
        diffsum = Float64(0)
        c = Int(0)

        #use a square Moore kernel
        for x in -1:1:1, y in -1:1:1

            #self-difference will be 0
            if x == 0 & y == 0
                continue
            end
            i1 = i - x
            j1 = j - y

            #don't go out-of-bounds
            if i1 < 1 || j1 < 1 || i1 > n || j1 > m
                continue
            end

            diffsum += vecnorm(tev[i,j] - tev[i1,j1])
            c += 1
        end
        diffs[i,j] = diffsum / c
    end

    return diffs
end

function te_stability(tev::Array{Array{Float64,2},2})
    diffs = te_differences(tev) .+ 1
    norms = vecnorm.(tev)
    return norms./diffs
end
