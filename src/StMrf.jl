module StMrf

import GCoptimization
GCO = GCoptimization;

import StMrfTracking.Tracking
import StMrfTracking.ImgBlock.Block

using PyCall

@pyimport skimage.feature as feature

const D_ROWS = [0 1 1 1 0 -1 -1 -1 0]
const D_COLS = [-1 -1 0 1 1 1 0 -1 0]

function is_foreground(block::Block, frame::Tracking.Img3Type, background::Tracking.Img3Type, threshold::Float64)
    y_inds = block.start_y:block.end_y
    x_inds = block.start_x:block.end_x
    
    return mean(abs.(frame[y_inds, x_inds, :] - background[y_inds, x_inds, :])) > threshold
end

function is_foreground(block::Block, foreground::BitArray{2})
    return mean(foreground[block.coords()...]) > 0.5
end

function update_slit_objects!(blocks::Array{Block, 2}, slit_coords, frame::Tracking.Img3Type, background::Tracking.Img3Type, 
                              new_block_id::Int; threshold::Float64=0.1)
    new_block_id = copy(new_block_id);
    obj_ids = zeros(Int, size(slit_coords, 1));

    for (id, coords) in enumerate(slit_coords)
        if !is_foreground(blocks[coords...], frame, background, threshold)
            continue
        end

        if blocks[coords...].object_id > 0
            obj_ids[id] = blocks[coords...].object_id
            continue
        end
        
        next_id = 0;
        for dc in zip(D_ROWS[1:end-1], D_COLS[1:end-1])
            new_coords = coords .+ dc
            if any(new_coords .< 1) || any(new_coords .> size(blocks))
                continue
            end
            
            next_id = blocks[new_coords...].object_id;
            if next_id != 0
                break
            end
        end
        
        if next_id != 0
            obj_ids[id] = next_id
            continue
        end
        
        if id > 1 && obj_ids[id - 1] > 0
            obj_ids[id] = obj_ids[id - 1];
            continue
        end

        obj_ids[id] = new_block_id
        new_block_id += 1
    end
    
    for (c, id) in zip(slit_coords, obj_ids)
        blocks[c...].object_id = id
    end
    
    return maximum(obj_ids) + 1
end

function motion_vector_diffs(blocks, frame, old_frame, coords)
    function get_diff(old_block_coords, new_block_coords)
        new_coords = blocks[new_block_coords...].coords();
        old_coords = blocks[old_block_coords...].coords();
        return sum(abs.(frame[new_coords...] .- old_frame[old_coords...]))
    end
    

    diffs = Array{Float64, 1}();
    for dc in zip(D_ROWS, D_COLS)
        cur_coords = coords .+ dc
        if any(cur_coords .< 1) || any(cur_coords .> size(blocks))
            cur_diff = 1e5
        else
            cur_diff = get_diff(coords, cur_coords)
        end
        push!(diffs, cur_diff)
    end
    
    return collect(zip(D_ROWS, D_COLS)), diffs
end

function motion_vector_similarity_map(blocks::Array{Block, 2}, frame::Tracking.Img3Type, 
                                      old_frame::Tracking.Img3Type, coords::Tuple{Int, Int}; 
                                      search_rad::Int=1)::Array{Float64, 2}
    if any(coords .- search_rad .< 1) || any(coords .+ search_rad .> size(blocks))
        res_size = (size(blocks[1].x_inds(), 1), size(blocks[1].y_inds(), 1)) .* 2 .* search_rad .- 1
        return zeros(res_size)
    end
    
    const x_coords = blocks[coords[1], coords[2] - search_rad].start_x:blocks[coords[1], coords[2] + search_rad].end_x;
    const y_coords = blocks[coords[1] - search_rad, coords[2]].start_y:blocks[coords[1] + search_rad, coords[2]].end_y;

    const cur_img_block::Tracking.Img3Type = frame[blocks[coords...].y_inds(), blocks[coords...].x_inds(),:];
    const match_img_region::Tracking.Img3Type = old_frame[y_coords, x_coords, :];

    return feature.match_template(match_img_region, cur_img_block)[:,:,1];
end

function find_motion_vector(blocks::Array{Block, 2}, frame::Tracking.Img3Type, old_frame::Tracking.Img3Type, 
                            group_coords::Array{Tuple{Int, Int}, 1}; search_rad::Int=1)
    const vec_diffs::Array{Array{Float64, 2}, 1} = [motion_vector_similarity_map(blocks, frame, old_frame, c, search_rad=search_rad) for c in group_coords];
    const sim_map = reduce(+, vec_diffs);
    const max_coords = collect(zip(findn(sim_map .== maximum(sim_map))...));
    const vecs = [c .- Int.((size(sim_map) .- 1) ./ 2) .- 1 for c in max_coords];
    return -1 .* vecs[findmin([sum(abs.(v)) for v in vecs])[2]]
end

function round_motion_vector(motion_vec::Union{Array{Int, 1}, Tuple{Int, Int}}, block_width::Int, block_height::Int)::Array{Int, 1}
    return round.(Int, motion_vec ./ [block_height, block_width])
end

function update_object_ids(blocks::Array{Block, 2}, block_id_map::Array{Int, 2}, 
                           motion_vecs::Union{Array{Tuple{Int, Int}, 1}, Array{Array{Int, 1}, 1}}, 
                           group_coords::Array{Array{Tuple{Int64,Int64},1},1}, frame::Tracking.Img3Type, 
                           background::Tracking.Img3Type; threshold::Float64=0.2)::Array{Set{Int}, 2}
    new_block_id_map::Array{Set{Int}, 2} = hcat([[Set{Int}() for _ in 1:size(block_id_map, 1)] for _ in 1:size(block_id_map, 2)]...)

    for (gc, m_vec) in zip(group_coords, motion_vecs)
        for coords in gc
            const new_coords = max.(min.(coords .+ m_vec, size(blocks)), [1, 1])

            if !is_foreground(blocks[new_coords...], frame, background, threshold)
                continue
            end

            const cur_block_id = block_id_map[coords...];
            push!(new_block_id_map[new_coords...], cur_block_id)

            for dc in zip(D_ROWS[1:end-1], D_COLS[1:end-1])
                const cur_coords = new_coords .+ dc
                if any(cur_coords .< 1) || any(cur_coords .> size(blocks))
                    continue
                end

                if cur_block_id in new_block_id_map[cur_coords...]
                    continue
                end

                if is_foreground(blocks[cur_coords...], frame, background, threshold)
                    push!(new_block_id_map[cur_coords...], cur_block_id)
                end
            end
        end
    end

    return new_block_id_map;
end

end