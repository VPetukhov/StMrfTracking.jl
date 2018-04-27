module Labeler

import GCoptimization
GCO = GCoptimization;

import VehicleTracker.GcWrappers
GW = GcWrappers;

import VehicleTracker.Tracking
import VehicleTracker.ImgBlock.Block

function unary_penalties(blocks, object_ids, motion_vecs, group_coords, prev_pixel_map, frame, prev_frame; inf_val::Float64=1000.0, mult::Float64=1000.0)
    unary_penalties = fill(inf_val, (length(blocks), size(group_coords, 1) + 1));
    unary_penalties[:, end] = 0;

    for (group_id, (obj_id, vec, gc)) in enumerate(zip(object_ids, motion_vecs, group_coords))
        for coords in gc
            block_id = GW.id_by_coords(coords..., size(blocks, 1), zero_based=false)

            block = blocks[coords...];

            cur_colors = frame[block.coords()..., :][:];
            prev_coords = block.coords(-1 .* vec);
            if any([prev_coords[1][end] prev_coords[2][end]] .> size(frame)) || any([prev_coords[1][1] prev_coords[2][1]] .< 0)
                unary_penalties[block_id, group_id] = 0 # TODO: process this case more cleverly
            else
                prev_colors = prev_frame[prev_coords..., :][:]
                img_diff = 1
                if std(cur_colors) == 0 || std(prev_colors) == 0
                    img_diff -= mean(abs.(cur_colors .- prev_colors))
                else
                    img_diff -= cor(cur_colors, prev_colors)
                end

                lab_diff = mean(prev_pixel_map[prev_coords...] .!= obj_id)

                unary_penalties[block_id, group_id] = img_diff + lab_diff
            end
            
            unary_penalties[block_id, end] = inf_val
        end
    end

    return -round.(Int, mult .* unary_penalties);
end

function label_map_naive(object_map::Array{Set{Int64},2})::Array{Int, 2}
    return map(ids -> length(ids) == 0 ? 0 : collect(ids)[1], object_map)
end

function label_map_gco(blocks::Array{Block, 2}, object_map::Array{Set{Int64},2}, motion_vecs::Array{Tuple{Int64,Int64},1}, 
                       prev_pixel_map::Array{Int, 2}, frame::Tracking.Img3Type, prev_frame::Tracking.Img3Type)::Array{Int, 2}
    if maximum(map(length, object_map)) < 2
        return label_map_naive(object_map)
    end

    const object_ids = sort(collect(reduce(union, object_map)));
    const group_coords = [collect(zip(findn(map(v -> lab in v, object_map))...)) for lab in object_ids];
    const data_cost = unary_penalties(blocks, object_ids, motion_vecs, group_coords, prev_pixel_map, frame, prev_frame);

    gco = GW.gc_optimization_8_grid_graph(size(blocks, 1), size(blocks, 2), size(data_cost, 2));
    GW.set_data_cost(gco, -data_cost)
    GCO.gco_setsmoothcost(gco, GW.smooth_cost_matrix(size(data_cost, 2), 1)) # TODO: set to 20
    GCO.gco_expansion(gco)
    
    const labels_by_ids = vcat(object_ids, [0]);
    const label_ids = GCO.gco_getlabeling(gco);
    return reshape(labels_by_ids[label_ids], size(blocks))
end
end