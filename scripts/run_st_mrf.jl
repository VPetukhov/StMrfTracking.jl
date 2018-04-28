using ArgParse

args = ArgParseSettings()
@add_arg_table args begin
    "--max-frames", "-m"
        help = "maximal number of frames to process in the file"
        arg_type = Int
        default = -1
    "--config", "-c"
        help = "JSON file with config"
        arg_type = String
        required = true
    "--suppress-shadows", "-s"
        help = "Suppress shadows"
        action = :store_true
    "--quite", "-q"
        help = "Don't show calibration image"
        action = :store_true
    "video_file"
        help = "video file to process"
        required = true
end

args = parse_args(args)

println("Loading modules... ")

import StMrfTracking

using ProgressMeter
using PyCall
using PyPlot
import Images

@pyimport imageio

T = StMrfTracking.Tracking;
SM = StMrfTracking.StMrf;
L = StMrfTracking.Labeler;
B = StMrfTracking.ImgBlock;

println("Done.")

# Parameters
const video_file = args["video_file"]
max_out_frames = args["max-frames"]
config = JSON.parsefile(args["config"]);
const suppress_shadows = args["suppress-shadows"]
const quite = args["quite"]

const slit_x = config["slit"]["x"];
const slit_y = config["slit"]["y"];
const slit_width = config["slit"]["width"];
const vehicle_direction = config["slit"]["vehicle_direction"];

const frame_step = config["video"]["frame_step"];
const n_background_init_frames = config["video"]["n_background_init_frames"];
const n_background_init_iters = config["video"]["n_background_init_iters"];

const block_width = config["blocks"]["width"];
const block_height = config["blocks"]["height"];

const threshold = config["algorithm"]["background_diff_threshold"];

# Algorithm
reader = imageio.get_reader(video_file);
if max_out_frames <= 0
    max_out_frames = reader[:get_meta_data]()["nframes"]
end

if !quite
    T.plot_frame(T.preprocess_frame(reader[:get_data](0)), slit_x, slit_y, slit_width, block_height)
    PyPlot.show()
end

print("Start reading $video_file... ")

const frames = frames = T.read_all_data(reader; frame_step=frame_step, max_frames=max_out_frames);

println("Done.")
print("Background initialization... ")

background = T.init_background(frames[1:min(n_background_init_frames, size(frames, 1))]; max_iters=n_background_init_iters);

println("Done.")

blocks = B.init_field(size(background)[1:2], (block_height, block_width))

slit_coords = [B.get_block_coords(slit_y, slit_x + i * block_width, block_height, block_width) for i in 1:floor(Int, slit_width / block_width)]
slit_line = [blocks[c...] for c in slit_coords];

const out_suffix = suppress_shadows ? ".out_sh" :  ".out";
out_video_file = join(splitext(video_file), out_suffix);
writer = imageio.get_writer(out_video_file, fps=reader[:get_meta_data]()["fps"] / frame_step);

println("Start video processing...")
new_object_id = 1;

@showprogress 1 "Processing video..." for f_id in 1:(size(frames, 1) - 1)
    frame = frames[f_id];
    prev_pixel_map = B.blocks_to_object_map(blocks);
    SM.update_slit_objects!(blocks, slit_coords, frame, background, new_object_id; threshold=threshold);
    
    old_frame = frame;
    frame = frames[f_id + 1];

    obj_ids_map = map(b -> b.object_id, blocks);

    object_ids = sort(unique(obj_ids_map));
    object_ids = object_ids[object_ids .> 0]
    group_coords = [collect(zip(findn(obj_ids_map .== id)...)) for id in object_ids]
    motion_vecs = [SM.find_motion_vector(blocks, frame, old_frame, gc) for gc in group_coords]
    
    labels = zeros(Int, size(blocks))
    if size(motion_vecs, 1) != 0
        motion_vecs_rounded = [SM.round_motion_vector(mv, block_width, block_height) for mv in motion_vecs]
        foreground = T.subtract_background(frame, background, threshold)

        if suppress_shadows
            foreground = T.suppress_shadows(foreground, frame, background)
        end
        new_map = SM.update_object_ids(blocks, obj_ids_map, motion_vecs_rounded, group_coords, foreground);

        if vehicle_direction == "up"
            new_map[map(x -> x.end_y, blocks) .> slit_y + block_height] = Set();
        else
            new_map[map(x -> x.end_y, blocks) .< slit_y] = Set();
        end

        labels = L.label_map_gco(blocks, new_map, motion_vecs, prev_pixel_map, frame, old_frame);
        labels = Images.label_components(labels, trues(3, 3))
    end
    
    new_object_id = maximum(labels) + 1
    B.set_block_ids!(blocks, labels)
    
    obj_map = B.blocks_to_object_map(blocks)
    out_frame = T.plot_frame_to_array(frame, slit_x, slit_y, slit_width, block_height; object_map=obj_map, plot_boxes=true)
    writer[:append_data](out_frame)
end

writer[:close]();
println("All done!");
println("Output file: $out_video_file");