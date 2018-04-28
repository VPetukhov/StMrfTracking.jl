module Tracking

import Images
import PyPlot
import StatsBase

using PyCall

@pyimport skimage.transform as transform
@pyimport skimage.filters as filters
@pyimport skimage.morphology as morphology
@pyimport skimage.color as color
@pyimport matplotlib.patches as patch
disk = morphology.disk;

Img2Type = Union{Array{UInt8,2}, Array{Float64,2}};
# Img3Type = Union{Array{UInt8,3}, Array{Float64,3}};
# ImgType = Union{Img2Type, Img3Type}
Img3Type = Array{Float64,3};

ImgArrType = Union{Array{Array{UInt8, 2}, 1}, Array{Array{Float64, 2}, 1}, 
                   Array{Array{UInt8, 3}, 1}, Array{Array{Float64, 3}, 1}};

const D_ROWS = [ 0  1 1 1 0 -1 -1 -1]
const D_COLS = [-1 -1 0 1 1  1  0 -1]

function read_all_data(reader; frame_step::Int=1, max_frames::Int=-1)
    if max_frames <= 0
        max_frames = reader[:get_meta_data]()["nframes"]
    end

    frames = Array{Img3Type, 1}()
    for i in 0:frame_step:(max_frames - 1)
        try
            push!(frames, preprocess_frame(reader[:get_data](i)))
        catch
            break
        end
    end

    return frames
end

function edge_image(img::Img2Type; g_max::Float64 = 256.0, 
        α::Float64 = 80.0, β::Float64 = 0.02)
    res = zeros(size(img));

    for i in 1:size(img, 1)
        i_min = max(1, i - 1);
        i_max = min(size(img, 1), i + 1);
        for j in 1:size(img, 2)
            j_min = max(1, j - 1);
            j_max = min(size(img, 2), j + 1);

            s = sum(abs.(img[i_min:i_max, j_min:j_max] .- img[i, j]))
            m = maximum(img[i_min:i_max, j_min:j_max])
            u = s / (m / g_max)
            res[i, j] = g_max / (1.0 + exp(-β*(u - α)))
        end
    end
    
    return res
end

function init_background_weighted(frames::ImgArrType; background_rate::Float64 = 0.95)
    background = zeros(size(frames[1]));

    for frame in frames
        background = background_rate .* background .+ (1 - background_rate) .* frame
    end
    return background
end

function init_background(frames::ImgArrType; background_rate::Float64 = 0.95, max_iters::Int=3, threshold_values::Array{Float64, 1} = [0.2; 0.1; 0.05])
    background = init_background_weighted(frames);

    for thres in threshold_values[1:max_iters]
        for (i, frame) in enumerate(frames)
            foreground = subtract_background(frame, background, thres);
            foreground = filters.median(foreground, disk(5)) .> 0
            update_background!(background, frame, foreground, background_rate);
        end
    
        for (i, frame) in enumerate(reverse(frames))
            foreground = subtract_background(frame, background, thres);
            foreground = filters.median(foreground, disk(5)) .> 0
            update_background!(background, frame, foreground, background_rate);
        end
    end
    
    return background;
end

function update_background!(background::Img3Type, frame::Img3Type, foreground::BitArray, background_rate::Float64=0.95)
    background_mask = .!foreground
    for i in 1:3
        cur_background = view(background, :,:,i)
        cur_frame = view(frame, :,:,i)
        cur_background[background_mask] .= background_rate .* cur_background[background_mask] .+ 
            (1 - background_rate) .* cur_frame[background_mask];
    end
end

function subtract_background(image::Img3Type, background::Img3Type, threshold::Float64=0.1)
    return any(abs.(image .- background) .> threshold, 3)[:,:,1]
end

function preprocess_frame(frame; shape::Array{Int, 1} = [480; 600], filt_radius::Int=2)::Img3Type
    img = transform.resize(frame, [480; 600]);
    if filt_radius == 0
        return img
    end
    return cat(3, [filters.median(img[:,:,i], disk(filt_radius)) for i in 1:3]...) ./ 255;
end

function plot_frame(frame, slit_x, slit_y, slit_width, block_height; object_map=nothing, plot_mask=false, plot_boxes=false, ax=nothing)
    if typeof(ax) == Void
        ax = PyPlot.axes()
    end
    if typeof(object_map) != Void && plot_mask
        frame = deepcopy(frame)
        for obj_id in unique(object_map)
            if obj_id == 0
                continue
            end
        
            srand(obj_id)
        
            mask = object_map .== obj_id;
            view(frame, :, :, 1)[mask] .= rand()
            view(frame, :, :, 2)[mask] .= rand()
            view(frame, :, :, 3)[mask] .= rand()
        end
    end

    ax[:imshow](frame)
    ax[:hlines](slit_y, xmin=slit_x, xmax=slit_x + slit_width)
    ax[:hlines](slit_y + block_height, xmin=slit_x, xmax=slit_x + slit_width);

    if typeof(object_map) != Void && plot_boxes
        b_boxes = Images.component_boxes(object_map)[2:end];
        for coords in b_boxes
            rect = patch.Rectangle(reverse(coords[1]), reverse(coords[2] .- coords[1])...)
            rect[:set_facecolor]("none")
            rect[:set_edgecolor]("red")
            ax[:add_artist](rect);
        end
    end
end

py"""
import numpy as np
def fig_to_array(fig):
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype="uint8")
    return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
"""

fig_to_array = py"fig_to_array";

function plot_frame_to_array(frame, slit_x, slit_y, slit_width, block_height; object_map=nothing, plot_mask=false, plot_boxes=false)
    fig = PyPlot.plt[:figure](figsize=size(frame)[[2, 1]] ./ 100, dpi=100)
    ax = fig[:gca]()
    plot_frame(frame, slit_x, slit_y, slit_width, block_height; object_map=object_map, plot_boxes=plot_boxes, plot_mask=plot_mask);
    ax[:axis]("off")
    fig[:subplots_adjust](left=0, bottom=0, top=1, right=1)
    arr = fig_to_array(fig)
    close(fig)
    return arr
end

function _find_adjacent_foreground_ratio(shadow_labels, foreground)
    const n_labels = maximum(shadow_labels)

    border_per_label = zeros(Int, n_labels);
    adj_foreground_per_label = zeros(Int, n_labels);
    pixels_per_class = zeros(Int, n_labels);

    for row in 1:size(shadow_labels, 1)
        for col in 1:size(shadow_labels, 2)
            const cur_class = shadow_labels[row, col]
            if cur_class == 0
                continue
            end

            pixels_per_class[cur_class] += 1;
            is_border = false
            for (dr, dc) in zip(D_ROWS, D_COLS)
                row_n = row + dr
                col_n = col + dc
                if row_n < 1 || col_n < 1 || row_n > size(shadow_labels, 1) || col_n > size(shadow_labels, 2)
                    continue
                end
                
                if shadow_labels[row_n, col_n] == cur_class
                    continue
                end
                
                if !is_border
                    is_border = true
                    border_per_label[cur_class] += 1
                end
                
                if foreground[row_n, col_n] && (shadow_labels[row_n, col_n] == 0)
                    adj_foreground_per_label[cur_class] += 1
                    break
                end
            end
        end
    end

    return adj_foreground_per_label ./ border_per_label, pixels_per_class
end

function segment_shadows(frame, shadow_mask; edge_threshold::Float64=0.05)
    shadow_mask = deepcopy(shadow_mask)

    edges = filters.sobel(mean(frame, 3)[:,:,1]) .> edge_threshold;
    edges[.!shadow_mask] = false
    shadow_mask[edges] = false;
    shadow_labels = Images.label_components(shadow_mask);
    next_label = maximum(shadow_labels) + 1;

    for coords in zip(findn(edges)...)
        adj_classes = Dict{Int, Int}()
        for dc in zip(D_ROWS, D_COLS)
            coords_n = coords .+ dc
            if any(coords_n .< 1) || any(coords_n .> size(shadow_labels))
                continue
            end
            
            const cur_label = shadow_labels[coords_n...];
            if cur_label == 0
                continue
            end
            
            adj_classes[cur_label] = get(adj_classes, cur_label, 0) + 1
        end
        
        if length(adj_classes) == 0
            shadow_labels[coords...] = next_label
            next_label += 1
        else
            shadow_labels[coords...] = collect(keys(adj_classes))[findmax(values(adj_classes))[2]]
        end
    end

    return shadow_labels
end

function estimate_ratio_score(foreground, shadow_foreground, frame, background)
    frame_ratio = (frame + 1 / 256) ./ (background + 1 / 256);
    ratio_pixels_shadow = hcat([frame_ratio[:,:,i][shadow_foreground] for i in 1:3]...);
    ratio_pixels = hcat([frame_ratio[:,:,i][foreground] for i in 1:3]...);

    const mean_est = median(ratio_pixels_shadow, 1);
    const std_est = mapslices(p -> StatsBase.mad(p; normalize=true), ratio_pixels, 1);

    return (ratio_pixels .- mean_est) ./ std_est
end

function estimate_shadow_mask(foreground, frame, ratio_score, z_threshold, adj_threshold, min_adj_threshold, size_threshold)
    foreground = deepcopy(foreground)
    shadow_mask_flatten = all(abs.(ratio_score) .< z_threshold, 2);
    shadow_mask = falses(size(frame)[1:2]);
    shadow_mask[foreground] .= vec(shadow_mask_flatten);

    fg_labels = Images.label_components(foreground .& .!shadow_mask) .+ 1;
    pixels_per_fg_label = zeros(Int, maximum(fg_labels));
    for lab in fg_labels
        pixels_per_fg_label[lab] += 1
    end

    size_border = (size_threshold * mean(pixels_per_fg_label[2:end]) + std(pixels_per_fg_label[2:end]))
    false_fg_mask = (pixels_per_fg_label .< size_border)[fg_labels]
    foreground[false_fg_mask] = false

    shadow_mask = filters.median(shadow_mask, disk(3)) .> 0;
    foreground = filters.median(foreground, disk(3)) .> 0;

    shadow_labels = segment_shadows(frame, shadow_mask);
    adj_foreground_ratio, pixels_per_bg_label = _find_adjacent_foreground_ratio(shadow_labels, foreground)

    is_shadow = (adj_foreground_ratio .< adj_threshold);
    is_shadow .&= (pixels_per_bg_label .< size_border) .| (adj_foreground_ratio .> min_adj_threshold)
    shadow_mask = vcat(false, is_shadow)[shadow_labels .+ 1];
    return shadow_mask
end

function suppress_shadows(foreground, frame, background; z_threshold::Number=1.0, adj_threshold::Float64=0.5, min_adj_threshold::Float64=0.1,
                          size_threshold::Number=0.5, shadow_diff_threshold::Float64=0.2, high_foreground_threshold::Float64=0.2)
    shadow_foreground = subtract_background(frame, background, high_foreground_threshold)
    ratio_score = estimate_ratio_score(foreground, shadow_foreground, frame, background)
    # ratio_score = estimate_ratio_score(foreground, foreground, frame, background)
    shadow_mask = estimate_shadow_mask(foreground, frame, ratio_score, z_threshold, adj_threshold, min_adj_threshold, size_threshold)

    shadow_foreground .&= shadow_mask

    ratio_score = estimate_ratio_score(foreground, shadow_foreground, frame, background)
    shadow_mask = estimate_shadow_mask(foreground, frame, ratio_score, z_threshold, adj_threshold, min_adj_threshold, size_threshold)

    foreground = filters.median(foreground .& .!shadow_mask, disk(5)) .> 0;

    return foreground
end

end
