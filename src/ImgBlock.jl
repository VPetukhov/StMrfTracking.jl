module ImgBlock

export Block, blocks_to_object_map, set_block_ids!, init_field

mutable struct Block
    start_x::Int;
    start_y::Int;
    end_x::Int;
    end_y::Int;
    object_id::Int;
    x_inds::Function;
    y_inds::Function;
    coords::Function;
    
    function Block(start_x::Int, start_y::Int, width::Int, height::Int)
        this = new(start_x, start_y, start_x + width, start_y + height, 0);
        
        this.x_inds = function() 
            return this.start_x:this.end_x
        end
        
        this.y_inds = function() 
            return this.start_y:this.end_y
        end
        
        this.coords = function(displacement::Union{Array{Int, 1}, Tuple{Int, Int}} = (0, 0)) 
            return (this.y_inds() .+ displacement[1], this.x_inds() .+ displacement[2])
        end
        
        return this
    end
end

function blocks_to_object_map(blocks::Array{Block, 2})
    end_x = blocks[1, end].end_x
    end_y = blocks[end, 1].end_y
    
    frame = zeros(Int, end_y, end_x)
    for block in blocks
        frame[block.coords()...] = block.object_id
    end
    
    return frame
end

function set_block_ids!(blocks::Array{Block, 2}, id_map::Array{Int, 2})
    if any(size(blocks) .!= size(id_map))
        error("Dimensions don't match: $(size(blocks)), $(size(id_map))")
    end

    for (b, id) in zip(blocks, id_map)
        b.object_id = id
    end
end

function get_block_coords(y::Int, x::Int, block_height::Int, block_width::Int)
    return (floor(Int, y / block_height), floor(Int, x / block_width))
end

function init_field(field_size::Union{Array{Int, 1}, Tuple{Int, Int}}, block_size::Union{Array{Int, 1}, Tuple{Int, Int}})
    blocks = Array{Block, 2}(floor.(Int, field_size ./ block_size) .- 1)
    for row_id in 1:size(blocks, 1)
        const row_start = row_id * block_size[1];
        for col_id in 1:size(blocks, 2)
            const col_start = col_id * block_size[2];
            blocks[row_id, col_id] = Block(col_start, row_start, block_size...);
        end
    end

    return blocks
end

end