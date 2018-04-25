module Tracking

import PyPlot
using PyCall

@pyimport skimage.transform as transform
@pyimport skimage.filters as filters
@pyimport skimage.morphology as morphology
@pyimport skimage.color as color
disk = morphology.disk;

Img2Type = Union{Array{UInt8,2}, Array{Float64,2}};
# Img3Type = Union{Array{UInt8,3}, Array{Float64,3}};
# ImgType = Union{Img2Type, Img3Type}
Img3Type = Array{Float64,3};

ImgArrType = Union{Array{Array{UInt8, 2}, 1}, Array{Array{Float64, 2}, 1}, 
                   Array{Array{UInt8, 3}, 1}, Array{Array{Float64, 3}, 1}};

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

function preprocess_frame(frame; shape::Array{Int, 1} = [480; 600], filt_radius::Int=2)
    img = transform.resize(frame, [480; 600]);
    return cat(3, [filters.median(img[:,:,i], disk(filt_radius)) for i in 1:3]...) ./ 255;
end

function plot_frame(frame, slit_x, slit_y, slit_width, block_height)
    PyPlot.imshow(frame)
    PyPlot.hlines(slit_y, xmin=slit_x, xmax=slit_x + slit_width)
    PyPlot.hlines(slit_y + block_height, xmin=slit_x, xmax=slit_x + slit_width);
end

function shadow_mask(frame, background; min_ratio::Float64=0.1, max_ratio::Float64=0.5, 
                     min_s::Float64=0.05, min_h::Float64=0.5)::BitArray{2}
    hsv_frame = color.rgb2hsv(frame);
    hsv_backround = color.rgb2hsv(background);

    h_mask = abs.(hsv_frame[:,:,1] .- hsv_backround[:,:,1]) .>= min_h;
    s_mask = (hsv_frame[:,:,2] .- hsv_backround[:,:,2] .>= min_s);
    v_mask = (min_ratio .<= (hsv_frame[:,:,3] ./ hsv_backround[:,:,3]) .<= max_ratio)

    return h_mask .& s_mask .& v_mask
end

end