[base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
max_frame = 294;
building_record = init_building_record(max_frame * 10);
for frame = 1 : max_frame
    building_instance = grab_instance_data(frame);
    included_building_instances = unique(building_instance(:));
    for j = 1 : length(included_building_instances)
        cur_instance = included_building_instances(j);
        if cur_instance == 0
            continue;
        end
        building_record{building_instance}(end + 1) = frame;
    end
end
empty_ind = find(~cellfun('isempty', building_record)); building_record(empty_ind) = [];
save('building_record.mat',building_record);

function building_record = init_building_record(max_frame)
    building_record = cell(max_frame, 1);
    for i = 1 : max_frame
        building_record{i} = generate_init_building_record_entry();
    end
end
function record_entry = generate_init_building_record_entry()
    record_entry = struct();
    record_entry.frames = zeros(0);
end
function building_instance = grab_instance_data(frame)
    [base_path, ~, ~, ~, GT_Color_Label_path, ~] = get_file_storage_path();
    f = num2str(frame, '%06d'); ImagePath = strcat(base_path, GT_Color_Label_path, f, '.png'); building_instance = imread(ImagePath);
end