function complot(result::Array{<:Real,2}, truth::Array{<:Real,2})
    n = size(result,1)
    matches = result .!= 0
    pcolormesh(truth .+ 0.5 .* (matches .!= truth),vmin=0, vmax=1.5)
    colorbar()
    text(n*1.11,n,"False +")
    text(n*1.11,n*0.70,"True +")
    text(n*1.11,n*0.33,"False -")
    text(n*1.11,n*0.05,"True -")
end

function te_sweep(recording::MEARecording, truth::Array{<:Real,2}, min::Real, max::Real, samples::Int, delays::Array{Int,1})
    tms = logspace(log(10,min),log(10,max),samples)
    n = length(delays)

    correct = zeros(length(tms), n, 2)

    for (i,bin) in enumerate(tms), delay in delays
        #println("BIN: ",bin," DELAY: ",delay)
        #discovered adjacency matrix
        teadj = edge_filter(full_TE(recording, bin, delay, 0), 3.0) .> 0
        #the number correct / total (acc)
        correct[i,delay,1] = sum(teadj .== truth)/length(truth)
        #the proportion of true positives found
        correct[i,delay,2] = sum(teadj .* truth)/sum(truth)
    end

    for i in 1:n
        semilogx(tms,correct[:,i,1],label=string("Acc., delay ",i))
        semilogx(tms,correct[:,i,2],label=string("Prec., delay ",i))
    end
    xlabel("TE. time bin size (s)")
    ylabel("Accuracy/Precision")
    legend()

    return correct
end

function delay_plot(x::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, i::Int, j::Int)
    result = x[1]
    delays = x[2]
    n = size(delays,1)

    plot(delays[:,i,j])
    xlabel("Bin Offset")
    ylabel("Transfer Entropy (bits)")
    plot(collect(1:n),repmat([result[2,i,j]+3*result[3,i,j]],n))
end

"""
    function connectionMap(mat)
Plot an adjacency matrix, showing white where no significant connection exists.
"""
function connectionMap(mat)
    colormap = matplotlib[:cm][:ScalarMappable](cmap="plasma");
    colors = colormap[:to_rgba](mat);

    fig = imshow(mat,cmap="plasma",interpolation="none")
    colorbar(fig)

    for i in 1:size(mat,1)
        for j in 1:size(mat,2)
            if mat[i,j] == 0.0
                #re-cast no connection to white
                colors[i,j,:] = [1.0,1.0,1.0, 1.0]
            end
        end
    end

    imshow(colors,interpolation="None")

end

"""
    function plotNormPMF(norms::Array{Float64,2}, timespace::Array{Float64,1})
Plot an image which represents the distribution of matrix norms using different time-limit parameters for analysis.
Darker colored regions indicate more matrices existing in that region.
Plots both norms and the velocity of changes in the norm.
"""
function plotNormPMF(norms::Array{Float64,2}, timespace::Array{Float64,1})
    #compute the diffs of the norms to plot velocity
    normdiffs = diff(norms,2)

    #compute the first, second, and third quartiles for each time step
    npercs = map(y->mapslices(x->percentile(x,y),norms,1),[25,50,75])
    #compute the full spectrum of percentiles 1-99 for each time step
    npercs_full = map(y->mapslices(x->percentile(x,y),norms,1),1:1:99)
    #do the same for the diffs
    dpercs = map(y->mapslices(x->percentile(x,y),normdiffs,1),[25,50,75])
    dpercs_full = map(y->mapslices(x->percentile(x,y),normdiffs,1),1:1:99)

    #create a figure
    figure(figsize=(12,5))
    #alpha step controls the intensity of each 'slice', lower means each will be fainter
    alphastep = 0.03
    subplot(121)
    #plot the median as a black line
    semilogx(timespace.*1000,npercs[2][1,:], "black",label="Median")

    for i in 1:49
        #for each percentile slice, fill between regions of equal probability (2%-98%, etc.)
        fill_between(timespace.*1000,npercs_full[50-i][1,:],npercs_full[50+i][1,:],facecolor="r",alpha=alphastep)
    end

    ylabel("Adj. Matrix Frobenius Norm")
    xlabel("Sampling window size (ms)")
    legend(loc="upper left")

    subplot(122)
    #make another plot, but plotting the distribution of changes in the matrix (velocities)
    vels = (diff(timespace)*1000)
    #normalize by dividing by the size of time diffs. this must be done because each step can be different if a log-space is being used.
    #plot the median as a black line
    semilogx(timespace[1:end-1].*1000.+vels.*0.5,dpercs[2][1,:]./vels, "black",label="Median")

    for i in 1:49
        #for each percentile slice, fill between regions of equal probability (2%-98%, etc.)
        fill_between(timespace[1:end-1].*1000.+vels.*0.5,dpercs_full[50-i][1,:]./vels,dpercs_full[50+i][1,:]./vels,facecolor="r",alpha=alphastep)
    end

    xlabel("Sampling window size (ms)")
    ylabel(L"Frob. Norm Velocity (ms$^{-1}$)")

    legend(loc="upper left")
end

"""
    pcaPlotvsDays(matrix::Array{Float64,2}, comp1::Int64, days::Array{Int64,1})
Generate a scatter plot of principal components of the given matrix.
Plots a component versus the days.
"""
function pcaPlotVsDays(matrix::Array{Float64,2}, comp1::Int64, days::Array{Int64,1})
    n = size(matrix,2) #the number of columns (variables)

    if comp1 > n
        throw(ArgumentError("Invalid principal component selection"))
    end

    (projection, relvar) = pca(matrix)

    colormap = matplotlib[:cm][:ScalarMappable](cmap="plasma")
    #colors = colormap[:to_rgba](days);
    scatter(days,projection[:,comp1])
    xlabel(string("Days in Vitro")); ylabel(string("PC", comp1, ", ", string(relvar[comp1])[1:4],"%"))
end

"""
    pcaPlot(matrix::Array{Float64,2}, comp1::Int64, comp2::Int64)
Generate a scatter plot of principal components of the given matrix.
Comp1 and comp2 select which principal components will be plotted.
"""
function pcaPlot(matrix::Array{Float64,2}, comp1::Int64, comp2::Int64)
    n = size(matrix,2) #the number of columns (variables)

    if comp1 > n || comp2 > n
        throw(ArgumentError("Invalid principal component selection"));
    end

    (projection, relvar) = pca(matrix)

    colormap = matplotlib[:cm][:ScalarMappable](cmap="plasma");
    colors = colormap[:to_rgba](collect(1:size(matrix,1)));
    scatter(projection[:,comp1],projection[:,comp2],color=colors);
    xlabel(string("PC", comp1, ", ", string(relvar[comp1])[1:4],"%")); ylabel(string("PC", comp2, ", ", string(relvar[comp2])[1:4],"%"));
end

"""
    pcaPlotByDays(matrix::Array{Float64,2}, comp1::Int64, comp2::Int64, days::Array{Int64,1})
Generate a scatter plot of principal components of the given matrix.
Comp1 and comp2 select which principal components will be plotted.
Each point's color will map to the number of days in vitro (days arg).
"""
function pcaPlotByDays(matrix::Array{Float64,2}, comp1::Int64, comp2::Int64, days::Array{Int64,1})
    n = size(matrix,2) #the number of columns (variables)

    if comp1 > n || comp2 > n
        throw(ArgumentError("Invalid principal component selection"));
    end

    (projection, relvar) = pca(matrix)

    colormap = matplotlib[:cm][:ScalarMappable](cmap="plasma");
    colors = colormap[:to_rgba](days);
    scatter(projection[:,comp1],projection[:,comp2],color=colors);
    xlabel(string("PC", comp1, ", ", string(relvar[comp1])[1:4],"%")); ylabel(string("PC", comp2, ", ", string(relvar[comp2])[1:4],"%"));
end

"""
    pcaPlotPaths(matrix::Array{Float64,2}, comp1::Int64, comp2::Int64, days::Array{Int64,1})
Generate a scatter plot of principal components of the given matrix.
Comp1 and comp2 select which principal components will be plotted.
The path of each culture through time will be plotted.
"""
function pcaPlotPaths(matrix::Array{Float64,2}, comp1::Int64, comp2::Int64, days::Array{Int64,1})
    n = size(matrix,2); #the number of columns (variables)

    if comp1 > n || comp2 > n
        throw(ArgumentError("Invalid principal component selection"));
    end

    (projection, relvar) = pca(matrix)

    colormap = matplotlib[:cm][:ScalarMappable](cmap="plasma");
    colors = colormap[:to_rgba](days);
    scatter(projection[:,comp1],projection[:,comp2],color=colors);
    xlabel(string("PC", comp1, ", ", string(relvar[comp1])[1:4],"%")); ylabel(string("PC", comp2, ", ", string(relvar[comp2])[1:4],"%"));

    #find the jumps in days to determine where plates change
    segments = find(diff(days) .< 0)
    segments = vcat(1, segments, length(days))

    for i = 1:length(segments)-1
        inds = segments[i]:segments[i+1]
        plot(projection[inds,comp1],projection[inds,comp2])
    end

end

"""
    plotLoadings(matrix::Array{Float64,2}, components::Array{Int64,1})
Plot the loadings of the given principal components across electrodes.
"""
function plotLoadings(matrix::Array{Float64,2}, components::Array{Int64,1})
    #plot the loadings of principal components to original variables
    loadings = pca(matrix)[3]

    figure();
    x = collect(1:size(loadings,1))

    for component in components
        if component > size(matrix,2) || component < 1
            println(string("Invalid component selection: ", component));
            continue;
        end

        plot(x, loadings[:,component], label=string("PCA",component));
    end
    legend(loc="best");
end

function comparison_plot(rec::StimMEARecording)
    se = stimulation_entropy(rec,0.100,5e-5)
    stirr = stirr(rec,0.100)

    figure(figsize=(15,4))
    subplot(131)
    imshow(se.>0,cmap="binary")
    title("Stimulation Entropy")

    subplot(132)
    imshow(stirr.>0,cmap="binary")
    title("StiRR")

    subplot(133)
    imshow((se.>0) .!= (stirr.>0),cmap="binary")
    title("Difference")
end

function ecdf(data::Array{<:Real,1}; label::String="")

    rangem = minimum(data)
    rangeM = maximum(data)

    ecdf(data, rangem, rangeM, label=label)
end

function ecdf(data::Array{<:Real,1}, rangem::Real, rangeM::Real; label::String="")

    n = 500
    ecdf = zeros(Float64,n)
    xsweep = linspace(rangem,rangeM,n)

    for (i,x) in enumerate(xsweep)
        greater = f->f<x
        ecdf[i] = length(find(greater, data))/length(data)
    end

    plot(xsweep,ecdf,label=label)
    ticklabel_format(scilimits=(0,-3))
    ylabel("ECDF")
    legend()

end

function compare_to_random_edges(x,y,f,n,d,af,c,lab)

    m = size(x,1)
    nodes = size(x[1],2)
    rand_agreement = zeros(Float64,n,m)

    @pyimport networkx as nx

    #for each repeat
    for i in 1:n
        #for each culture
        for j in 1:m
            xc = edge_count(x[j],f)
            yc = edge_count(y[j],f)
            num_edges = round(Int,min(xc,yc) + rand()*abs(xc-yc))
            #print(num_edges," ")

            g1 = nx.gnm_random_graph(nodes,num_edges,directed=true)
            net1 = nx.adj_matrix(g1)[:toarray]()
            g2 = nx.gnm_random_graph(nodes,num_edges,directed=true)
            net2 = nx.adj_matrix(g2)[:toarray]()

            rand_agreement[i,j] = af(net1,net2)

        end
        #println()
        if i == 1
            plot(d,rand_agreement[i,:],alpha=0.1,color=c,label=lab)
        else
            plot(d,rand_agreement[i,:],alpha=0.1,color=c)
        end
    end

end

function truth_ecdf(surrs, exp, adj)
#     means = squeeze(mean(surrs,1),1)
#     stds = squeeze(std(surrs,1),1)
    means = surrs[1]
    stds = surrs[2]

    trues = find(x->x==1,vec(adj))
    falses = find(x->x==0,vec(adj))

    t_sigs = filter(!isnan,(exp[trues] .- means[trues]) ./ stds[trues])
    f_sigs = filter(!isnan,(exp[falses] .- means[falses]) ./ stds[falses])
    rangem = min(minimum(t_sigs),minimum(f_sigs))
    rangeM = max(maximum(t_sigs),maximum(f_sigs))

    n_t = length(t_sigs)
    n_f = length(f_sigs)

    n = 500
    ecdf_t = zeros(Float64,n)
    ecdf_f = zeros(Float64,n)
    xsweep = linspace(rangem,rangeM,n)

    for (i,x) in enumerate(xsweep)
        greater = f->f<x
        ecdf_t[i] = length(find(greater, t_sigs))/n_t
        ecdf_f[i] = length(find(greater, f_sigs))/n_f
    end

    plot(xsweep,ecdf_f,label="Falses")
    plot(xsweep,ecdf_t,label="Trues")
    xlabel("Significance (sigma)")
    ylabel("ECDF")
    legend()

    return ((minimum(t_sigs) - maximum(f_sigs)), (median(t_sigs) - median(f_sigs)))
end
