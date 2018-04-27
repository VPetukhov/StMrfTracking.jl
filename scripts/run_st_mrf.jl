println("Loading modules... ")

using PyCall
using PyPlot
import Images

@pyimport imageio

using StMrfTracking
T = Tracking;
SM = StMrf;
L = Labeler;
B = ImgBlock;

println("Done.")

### TODO: move to config and cli arguments
const video_file = "/home/viktor/VirtualBox VMs/Shared/VehicleTracking/vehicle_videos/4K_p2.mp4"
const frame_step = 2;
const frames = [T.preprocess_frame(reader[:get_data](i)) for i in 0:frame_step:(n_frames-1)];
const n_background_init_frames = 300
const n_background_init_iters = 3

const slit_x = 50;
const slit_y = 400;
const slit_width = 250;

const block_width = 10;
const block_height = 8;

const threshold = 0.05;
max_out_frames = -1
###

reader = imageio.get_reader(video_file);
if max_out_frames <= 0
    max_out_frames = reader[:get_meta_data]()["nframes"]
end

print("Start reading $video_file... ")

frames = [T.preprocess_frame(reader[:get_data](i)) for i in 0:frame_step:(max_out_frames-1)];

println("Done.")
print("Background initialization... ")

background = T.init_background(frames[1:min(n_background_init_frames, size(frames, 1))]; max_iters=n_background_init_iters);

println("Done.")

blocks = B.init_field(size(background)[1:2], (block_height, block_width))

slit_coords = [B.get_block_coords(slit_y, slit_x + i * block_width, block_height, block_width) for i in 1:floor(Int, slit_width / block_width)]
slit_line = [blocks[c...] for c in slit_coords];

out_video_file = join(splitext(video_file), ".out");
writer = imageio.get_writer(out_video_file, fps=reader[:get_meta_data]()["fps"] / frame_step);

println("Start video processing...")
new_object_id = 1;
for f_id in 1:(size(frames, 1) - 1)
    if f_id % 100 == 0
        println("Total $f_id frames processed.")
    end
    
    global frame = frames[f_id];
    global prev_pixel_map = B.blocks_to_object_map(blocks);
    SM.update_slit_objects!(blocks, slit_coords, frame, background, new_object_id; threshold=threshold);
    
    global old_frame = frame;
    frame = frames[f_id + 1];

    obj_ids_map = map(b -> b.object_id, blocks);

    global object_ids = sort(unique(obj_ids_map));
    object_ids = object_ids[object_ids .> 0]
    group_coords = [collect(zip(findn(obj_ids_map .== id)...)) for id in object_ids]
    global motion_vecs = [SM.find_motion_vector(blocks, frame, old_frame, gc) for gc in group_coords]
    
    labels = zeros(Int, size(blocks))
    if size(motion_vecs, 1) != 0
        motion_vecs_rounded = [SM.round_motion_vector(mv, block_width, block_height) for mv in motion_vecs]
        global new_map = SM.update_object_ids(blocks, obj_ids_map, motion_vecs_rounded, group_coords, frame, background; 
                                       threshold=threshold);
#         new_map[map(x -> x.end_y, blocks) .< slit_y] = Set();
        new_map[map(x -> x.end_y, blocks) .> slit_y + block_height] = Set();

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