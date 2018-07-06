### NETWORK ANALYSIS FUNCTIONS ###

"""
Given a matrix representing experimental / mean / std values of the adjacency matrix,
return a matrix of the sigma value of each connection
"""
function edge_significance(mat::Array{Float64,3})
    return filter(!isnan, mat[1,:,:] .- (mat[2,:,:] ./ mat[3,:,:]))
end

"""
Given a matrix representing experimental / mean / std values of the adjacency matrix,
return a matrix of the experimental values above a sigma-based significance threshold.
"""
function edge_filter(mat::Array{Float64,3}, sig::Real)
    return (mat[1,:,:] .* (mat[1,:,:] .> (mat[2,:,:] .+ sig .* mat[3,:,:])))
end

"""
Given a matrix representing experimental / mean / std values of the adjacency matrix,
return the number of the experimental values above a sigma-based significance threshold.
"""
function edge_count(mat::Array{Float64,3}, sig::Real)
    return length(find(edge_filter(mat,sig).>0))
end


function nz(result::Array{<:Real,2})
    return result .!= 0.0
end

function tp(result::Array{<:Real,2}, truth::Array{<:Real,2})
    #where both truth and result are 1
    return sum(nz(result) .* truth)
end

"""
False negatives in an adjacency matrix compared to ground truth.
"""
function fn(result::Array{<:Real,2}, truth::Array{<:Real,2})
    #where truth is 1 and result is 0
    return sum(truth .* (nz(result) .== 0.0))
end

"""
False positives in an adjacency matrix compared to ground truth.
"""
function fp(result::Array{<:Real,2}, truth::Array{<:Real,2})
    #where truth is 0 and result is 1
    return sum((truth .== 0.0) .* (nz(result)))
end

"""
True negatives in an adjacency matrix compared to ground truth.
"""
function tn(result::Array{<:Real,2}, truth::Array{<:Real,2})
    #both truth and result are 0
    return sum((truth .== 0) .* (nz(result) .== 0))
end

"""
Precision of an adjacency matrix compared to ground truth.
"""
function mprecision(result::Array{<:Real,2}, truth::Array{<:Real,2})
    # TP / (TP + FP)
    return tp(result, truth)/sum(truth)
end

"""
Accuracy of an adjacency matrix compared to ground truth.
"""
function maccuracy(result::Array{<:Real,2}, truth::Array{<:Real,2})
    return sum(nz(result) .== truth)/length(truth)
end

"""
Recall of an adjacency matrix compared to ground truth.
"""
function mrecall(result::Array{<:Real,2}, truth::Array{<:Real,2})
    # TP / TP + FN
    x = tp(result, truth)
    y = fn(result, truth)
    return x / (x+y)
end

"""
Specificity of an adjacency matrix compared to ground truth.
"""
function mspecificity(result::Array{<:Real,2}, truth::Array{<:Real,2})
    x = tn(result, truth)
    y = fp(result, truth)
    return x / (x+y)
end

"""
F1 score of an adjacency matrix compared to ground truth.
"""
function f1(result::Array{<:Real,2}, truth::Array{<:Real,2})
    return 2/(1/mprecision(result,truth) + 1/mrecall(result,truth))
end

"""
Cross-accuracy between [exp / mu / std ] adjacency matrices, filtered by sigma value f.
"""
function xaccuracy(x::Array{<:Real,3},y::Array{<:Real,3},f::Real)
    x1 = edge_filter(x,f) .> 0
    y1 = edge_filter(y,f) .> 0

    return xaccuracy(x1,y1)
end

"""
Cross-accuracy between experimental adjacency matrices.
"""
function xaccuracy(x::Array{<:Real,2},y::Array{<:Real,2})
    consistencies = sum((x .+ y) .!= 1.0)

    return (consistencies) / length(x)
end

"""
Cross-precision between [exp / mu / std ] adjacency matrices, filtered by sigma value f.
"""
function xprecision(x::Array{<:Real,3},y::Array{<:Real,3},f::Real)
    x1 = edge_filter(x,f) .> 0
    y1 = edge_filter(y,f) .> 0

    return xprecision(x1,y1)
end


"""
Cross-precision between experimental adjacency matrices.
"""
function xprecision(x::Array{<:Real,2},y::Array{<:Real,2})
    tpos = sum((x .* y) .> 0)
    xp = sum(x)
    yp = sum(y)

    return (2*tpos)/(xp+yp)
end

"""
One-tailed permutation test between three-dimensional arrays representing distributions of values for adjacency matrices
"""
function permutation_test(experimental::Array{Float64,3}, null::Array{Float64,3}, n_p::Int; dist::Bool=false)
    if size(experimental)[2:3] != size(null)[2:3]
        error("Distributions must have equal second and third dimension")
    end

    s = size(experimental,2)
    n1 = size(experimental,1)
    n2 = size(null,1)
    t = n1 + n2

    median_reduce(x) = squeeze(median(x,1),1)

    #find the observed difference between the distributions
    t_obs = median_reduce(experimental) .- median_reduce(null)

    #create a list of labels which we can shuffle to select which distribution to draw from
    selection = zeros(Int64, t)
    for i in 1:n1
      selection[i] = 1
    end

    for i in (n1+1):t
      selection[i] = 2
    end

    exceeding = zeros(Int64,s,s)
    ind1 = Int(1)
    d1_i = collect(1:n1)
    ind2 = Int(1)
    d2_i = collect(1:n2)

    sums1 = zeros(Float64,s,s)
    sums2 = zeros(Float64,s,s)

    if dist
        tests = zeros(Float64,n_p,s,s)
    end
    #now do the permutation tests
    for p in 1:n_p
        #shuffle the data and find if its value is greater than the observed
        shuffle!(selection)
        shuffle!(d1_i)
        shuffle!(d2_i)

        #calculate the new average value shuffled lagbels
        #resample the first distribution (sums1)
        for i in 1:n1
          if selection[i] == 1
            sums1 .+= experimental[d1_i[ind1],:,:]
            ind1 += 1
          else
            sums1 .+= null[d2_i[ind2],:,:]
            ind2 += 1
          end
        end
        #resample the second distribution (sums2)
        for i in (n1+1):t
          if selection[i] == 1
            sums2 .+= experimental[d1_i[ind1],:,:]
            ind1 += 1
          else
            sums2 .+= null[d2_i[ind2],:,:]
            ind2 += 1
          end
        end

        #then take the average
        median1 = sums1 ./ n1
        median2 = sums2 ./ n2

        t_sample = median1 .- median2

        if dist
            tests[p,:,:] = t_sample
        end

        comparison = (t_sample .>= t_obs)

        #if the permutation value was greater than the experimental, add 1 to the counter
        exceeding[comparison] .+= 1

        #reset the sums / counters
        fill!(sums1, 0)
        fill!(sums2, 0)
        ind1 = 1
        ind2 = 1
    end

    #divide by the number of permutations to obtain the signifiance
    result = exceeding ./ n_p
    if dist
        return (result, tests)
    else
        return result
    end
end


"""
A Mann-Whitney U test between arrays of samples of adjacency matrix.
Returns the chance the experimental sample will be lesser than the surrogate value.
"""
function u_test(exp::Array{Float64,3}, surrs::Array{Float64,3}, minfilter::Real=0.0)
   dim = size(exp,2)
   n1 = size(exp,1)
   n2 = size(surrs,1)

   exceeding = zeros(Int,dim,dim)
   total = n1*n2
   #create a mask which will remove any elements which have a trivial TE value
   zero_mask = (squeeze(sum(exp,1),1) .== 0.0)
   #create a mask which will remove elements below the minimum threshold
   filter_mask = (squeeze(median(exp,1),1) .< minfilter)

   for p in 1:total
       i1 = mod1(p,n1)
       i2 = div(p-1,n1) + 1

       exceeding[surrs[i2,:,:] .> exp[i1,:,:]] .+= 1
   end

   results = (exceeding ./ total)
   results[zero_mask] .= 1.0
   results[filter_mask] .= 1.0
   return results
end

"""
Komologrov-Smirnov test of a sample against a fit Gaussian distribution.
"""
function KS1(dist1::Array{Float64,1})

    rangem = minimum(dist1)
    rangeM = maximum(dist2)
    sup = Float64(0)

    n = 200
    ecdf = zeros(Float64,n,2)
    xsweep = linspace(rangem,rangeM,n)
    sigma = std(dist1)
    mu = mean(dist1)
    gauss(x::Real) = (1/sqrt(2*pi*sigma^2))*exp(-1*(x-mu)^2/(2*sigma^2))
    dist2 = [gauss(x) for x in xsweep]

    for (i,x) in enumerate(xsweep)
        greater = f->f<x
        ecdf[i,1] = length(find(greater, dist1))/length(dist1)
        ecdf[i,2] = length(find(greater, dist2))/length(dist2)
        distance = abs(ecdf[i,1] - ecdf[i,2])
        if distance > sup
            sup = distance
        end
    end

#     plot(xsweep,ecdf)
#     ticklabel_format(scilimits=(0,-3))
#     ylabel("ECDF")
#     legend()

    return sup
end

"""
Komolgrov-Smirnov test of two samples against each other.
"""
function KS2(dist1::Array{Float64,1}, dist2::Array{Float64,1})

    range1 = (minimum(dist1), maximum(dist1))
    range2 = (minimum(dist2), maximum(dist2))

    rangem = min(range1[1],range2[1])
    rangeM = max(range2[2],range2[2])
    sup = Float64(0)

    n = 200
    ecdf = zeros(Float64,n,2)
    xsweep = linspace(rangem,rangeM,n)

    for (i,x) in enumerate(xsweep)
        greater = f->f<x
        ecdf[i,1] = length(find(greater, dist1))/length(dist1)
        ecdf[i,2] = length(find(greater, dist2))/length(dist2)
        distance = abs(ecdf[i,1] - ecdf[i,2])
        if distance > sup
            sup = distance
        end
    end

#     plot(xsweep,ecdf)
#     ticklabel_format(scilimits=(0,-3))
#     ylabel("ECDF")
#     legend()

    return sup
end

KSalpha(a::Real, n::Int, m::Int) = sqrt(-0.5*log(a/2))*sqrt((n+m)/(n*m))

# Network analysis features - disabled by default to avoid PyImport / networkx requirement.

# """
#     betweennessCentralities(rawTransfers::Array{Float64,2})
#
# Given an adjacency matrix and the configuration, determine the betweenness centrality of each node in the graph.
# Uses Python NetworkX.
# """
# function betweennessCentralities(rawTransfers::Array{Float64,2})
#     #return an array of the electrodes' betweenness centrality
#
#     if size(rawTransfers,1) != size(rawTransfers,2)
#         throw(ArgumentError("Input matrix must be square (an adjacency matrix)."))
#     end
#
#     @pyimport networkx as ntx2
#     transfers = copy(rawTransfers);
#     #convert values into absolute values, in case it's a mean transfer matrix
#     transfers = abs.(transfers);
#     #divide by the largest value to cast it into a 0-1 "adjacency" matrix
#     #transfers = transfers./maximum(transfers);
#
#     #convert that matrix into a directional graph
#     graph = ntx2.DiGraph(transfers);
#     #calculate the centralities for each node
#     centrality_dict = ntx2.betweenness_centrality(graph);
#
#     dim = size(rawTransfers,1)
#     centralities = zeros(dim)
#     #pull the values for each electrode out of the dictionary
#     #only lookup the values for active electrodes
#     for key in keys(centrality_dict)
#         #the vertices are zero-indexed
#         centralities[key+1] = centrality_dict[key]
#     end
#
#     return centralities;
# end
#
# """
#     degreeCentralities(rawTransfers::Array{Float64,2})
#
# Given an adjacency matrix and the configuration, determine the normalized degree of each node in the graph.
# Uses Python NetworkX.
# """
# function degreeCentralities(rawTransfers::Array{Float64,2},dir_in::Bool=false, dir_out::Bool=false)
#     #return an array of the electrodes' betweenness centrality
#
#     if size(rawTransfers,1) != size(rawTransfers,2)
#         throw(ArgumentError("Input matrix must be square (an adjacency matrix)."))
#     end
#
#     @pyimport networkx as ntx
#     transfers = copy(rawTransfers);
#     #convert values into absolute values, in case it's a mean transfer matrix
#     transfers = abs.(transfers);
#     #divide by the largest value to cast it into a 0-1 "adjacency" matrix
#     #transfers = transfers./maximum(transfers);
#
#     #convert that matrix into a directional graph
#     graph = ntx.DiGraph(transfers);
#     #calculate the centralities for each node
#     if dir_in
#         centrality_dict = ntx.in_degree_centrality(graph)
#     elseif dir_out
#         centrality_dict = ntx.out_degree_centrality(graph)
#     else
#         centrality_dict = ntx.degree_centrality(graph)
#     end
#
#     dim = size(rawTransfers,1)
#     centralities = zeros(dim)
#     #pull the values for each electrode out of the dictionary
#     #only lookup the values for active electrodes
#     for key in keys(centrality_dict)
#         #the vertices are zero-indexed
#         centralities[key+1] = centrality_dict[key]
#     end
#
#     return centralities;
# end
# """
# Find cross-accuracy/precision values for randomly-generated Erdos-Renyi graphs - represents null hypothesis
# """
# function compare_to_random_edges(size,edges,samples)
#     rand_acc = zeros(Float64,)
#     rand_prec = zeros(Float64,n,2)
#
#     m = size(arr[1],1)
#     nets = falses(samples,m,m)
#
#     @pyimport networkx as nx
#
#     #print("current ind: ")
#     for i in 1:n
#         print(i," ")
#         num_edges = sum(arr[i].>0)
#         for j in 1:samples
#             g = nx.gnm_random_graph(m,num_edges,directed=true)
#             nets[j,:,:] = nx.adj_matrix(g)[:toarray]()
#         end
#
#         accs = [matrixAccuracy(nets[j,:,:],arr[i]) for j in 1:samples]
#         precs = [matrixPrecision(nets[j,:,:],arr[i]) for j in 1:samples]
#         rand_acc[i,1] = mean(accs)
#         rand_acc[i,2] = std(accs)
#
#         rand_prec[i,1] = mean(precs)
#         rand_prec[i,2] = std(precs)
#     end
#
#     return (rand_acc, rand_prec)
# end
# function compare_to_random_edges(x,y,f,n)
#
#     m = size(x,1)
#     nodes = size(x[1],2)
#     rand_agreement = zeros(Float64,n,m)
#
#     @pyimport networkx as nx
#
#     #for each repeat
#     for i in 1:n
#         #for each culture
#         for j in 1:m
#             xc = edge_count(x[j],f)
#             yc = edge_count(y[j],f)
#             num_edges = round(Int,min(xc,yc) + rand()*abs(xc-yc))
#             #print(num_edges," ")
#
#             g1 = nx.gnm_random_graph(nodes,num_edges,directed=true)
#             net1 = nx.adj_matrix(g1)[:toarray]()
#             g2 = nx.gnm_random_graph(nodes,num_edges,directed=true)
#             net2 = nx.adj_matrix(g2)[:toarray]()
#
#             rand_agreement[i,j] = sum((net1 .+ net2) .!= 1.0)/length(net1)
#
#         end
#         #println()
#         if i == 1
#             plot(rand_agreement[i,:],alpha=0.2,color="orange",label="ER-Random Graph")
#         else
#             plot(rand_agreement[i,:],alpha=0.2,color="orange")
#         end
#     end
#
# end

### ANALYSIS FUNCTIONS (PCA) ###

"""
    pca(matrix::Array{Float64,2}

Custom PCA function to give a constistent orientation for principal components.
Returns tuple of projections and relative variances
"""
function pca(matrix::Array{Float64,2})
      #custom interal PCA function to orient components in a reliable manner for plotting
      column_mean = mean(matrix,1) #the per-variable means
      rows = size(matrix,1) #the number of samples (rows)
      n = size(matrix,2) #the number of columns (variables)

      #center the values around the means, by column (variable)
      matrix = matrix.-column_mean
      #scale the matrix so that the singular values look like the covariance matrix's eigenvalues
      matrix /= sqrt(rows - 1)

      #computer the singular value decomposition of the rescaled matrix
      (U, S, V) = svd(matrix)

      #make sure that the first few singular values are negative, and flip them if they aren't to get a constistent projection
      if (S[1]*U[1,1] > 0)
        S[1] *= -1
      end
      if (S[2]*U[1,2] > 0)
        S[2] *= -1
      end

      eigenvalues = S.^2
      relvar = 100*eigenvalues/sum(eigenvalues)

      projection = U*diagm(S)

      return (projection, relvar)
end


"""
Given an array of adjacency matrices and the corresponding days-in-vitro, return a matrix that represents the
day a connection emerged permanently (did not later disappear/reappear). 0 if did not exist.
"""
function permanent_emergence(adjs::Array{Array{Float64,2},1}, days::Array{Int,1}, sig::Real=0.01)
    n = length(adjs)
    if length(days) != n
        error("Length of days must match number of adjacency matrices.")
    end
    dim = size(adjs[1],1)
    e_ind = zeros(Int,dim,dim)

    for j in 1:dim, k in 1:dim
        i = n
        while adjs[i][j,k] < sig && i > 1
            i -= 1
        end
        if i != n
            i += 1
            e_ind[j,k] = days[i]
        end
    end

    return e_ind
end

"""
Given an array of adjacency matrices and the corresponding days-in-vitro, return a matrix that represents the
day a connection first emerged (may have later disappeared/reappeared). 0 if did not exist.
"""
function emergence(adjs::Array{Array{Float64,2},1}, days::Array{Int,1}, sig::Real=0.01)
    n = length(adjs)
    if length(days) != n
        error("Length of days must match number of adjacency matrices.")
    end
    dim = size(adjs[1],1)
    e_ind = zeros(Int,dim,dim)

    for i in 1:n
        for j in 1:dim, k in 1:dim
            if e_ind[j,k] == 0 && adjs[i][j,k] < sig
                e_ind[j,k] = days[i]
            end
        end
    end

    return e_ind
end
