module GcWrappers

using GCoptimization

function id_by_coords(row::Int, col::Int, height::Int; zero_based::Bool=true)
    id = [col - 1, row]' * [height, 1]

    if zero_based
        return id - 1
    end
    
    return id
end

function prob_to_score(prob::Float64; mult::Number=5)::Int
    return -round(Int, mult * log10(1e-20 + prob))
end

function gc_optimization_8_grid_graph(height::Int, width::Int, n_labels::Int)
    gco = GCoptimizationGeneralGraph(height * width, n_labels)

    for row in 1:(height - 1)
        for col in 1:(width - 1)
            setNeighbors(gco, id_by_coords(row, col, height), id_by_coords(row + 1, col, height))
            setNeighbors(gco, id_by_coords(row, col, height), id_by_coords(row, col + 1, height))
            setNeighbors(gco, id_by_coords(row, col, height), id_by_coords(row + 1, col + 1, height))
            setNeighbors(gco, id_by_coords(row + 1, col, height), id_by_coords(row, col + 1, height))
        end
    end
    
    return gco
end

function set_data_cost(gco, costs::Array{Int, 2})
    for node_id in 1:size(costs, 1)
        for label_id in 1:size(costs, 2)
            setDataCost(gco, node_id - 1, label_id - 1, costs[node_id, label_id])
        end
    end
end

function set_smooth_cost(gco, n::Int; penalty::Int=1)
    gco_setsmoothcost(gco, smooth_cost_matrix(n, penalty))
end

function smooth_cost_matrix(n::Int, penalty::Int=1)
    mtx = ones(Int, n, n) .- diagm(ones(Int, n));
    return mtx .* penalty
end

function sigmoid(x::Float64, μ::Float64=0.0; scale::Float64=1.0)
    return 1 / (1 + exp(scale .* (μ - x)))
end


function adj_smooth_cost(frame, coords1, coords2)
    return prob_to_score(1. - maximum(abs.(frame[coords1...,:] .- frame[coords2...,:])), mult=100)
end

function set_w_neighbors(gco, frame, coords1, coords2, height)
    const id1 = id_by_coords(coords1..., height)
    const id2 = id_by_coords(coords2..., height)
    setNeighbors(gco, id1, id2, adj_smooth_cost(frame, coords1, coords2))
end

end