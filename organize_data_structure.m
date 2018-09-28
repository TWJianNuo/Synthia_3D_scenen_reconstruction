% Get initial guess of the cubic shape
[base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
intrinsic_params = get_intrinsic_matrix(); init_entry_num = 20; data_cluster = cell(init_entry_num, 1); data_cluster_prop = init_data_cluster_prop();
n = 294; exp_re_path = make_dir();
for frame = 1 : n
    [extrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame);
    org_frame_data = read_in_org_entry(frame); [data_cluster, data_cluster_prop] = merge_obj_into_data_cluster(org_frame_data, data_cluster, data_cluster_prop);
    save_organized_data(exp_re_path, frame, data_cluster, data_cluster_prop);
end
function save_organized_data(path, frame, data_cluster, data_cluster_prop)
    if(format_check(data_cluster, data_cluster_prop))
        noempty_element = find(~cellfun(@isempty, data_cluster));
        data_cluster = data_cluster(noempty_element);
        data_cluster_prop.num_record = data_cluster_prop.num_record(noempty_element, :);
        save([path '/' num2str(frame, '%06d') '.mat'], 'data_cluster', 'data_cluster_prop')
    end
end
function is_valid = format_check(data_cluster, data_cluster_prop)
    real_num_record = zeros(size(data_cluster_prop.num_record));
    for i = 1 : data_cluster_prop.num
        for j = 1 : data_cluster{i}.num
            real_num_record(i, j) = length(data_cluster{i}.linear_ind{j});
        end
    end
    is_valid = isequal(real_num_record, data_cluster_prop.num_record);
    if ~is_valid
        disp('Data corrupted')
    end
end
function org_entry = read_in_org_entry(frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/20_Sep_2018_22_segmentation/Instance_map/';
    ind = num2str(frame, '%06d');
    loaded = load([path ind '.mat']); org_entry = loaded.prev_mark;
end
function data_cluster_prop = init_data_cluster_prop()
    data_cluster_prop.num = 0;
    data_cluster_prop.num_record = zeros(20, 20);
end
function [data_cluster, data_cluster_prop] = down_sample_data_cluster(data_cluster, data_cluster_prop)
    allowed_max_pt_num = 30000; cur_tot_pt_num = sum(sum(data_cluster_prop.num_record));
    if allowed_max_pt_num < cur_tot_pt_num
        sample_rate = allowed_max_pt_num / cur_tot_pt_num;
        data_cluster_prop.num_record = ceil(data_cluster_prop.num_record .* sample_rate);
        iterated_num = 1;
        while iterated_num <= data_cluster_prop.num
            [data_cluster{iterated_num}, data_cluster_prop.num_record(iterated_num,:)] = down_sample_entry(data_cluster{iterated_num}, data_cluster_prop.num_record(iterated_num,:));
            if sum(data_cluster_prop.num_record(iterated_num,:)) == 0
                data_cluster(iterated_num) = [];
                data_cluster_prop.num = data_cluster_prop.num - 1;
                data_cluster_prop.num_record(iterated_num, :) = [];
                iterated_num = iterated_num - 1;
            end
            iterated_num = iterated_num + 1;
        end
    end
end
function [data_cluster_entry, num_array] = down_sample_entry(data_cluster_entry, num_array)
    iterated_num = 1; min_entry_num = 200;
    while iterated_num <= data_cluster_entry.num
        cur_num = num_array(iterated_num);
        sampled_ind = ceil(linspace(1, length(data_cluster_entry.linear_ind{iterated_num}), cur_num));
        if length(sampled_ind) == 1
            data_cluster_entry.linear_ind(iterated_num) = [];
            data_cluster_entry.pts_old(iterated_num) = [];
            data_cluster_entry.pts_new(iterated_num) = [];
            data_cluster_entry.frame(iterated_num) = [];
            data_cluster_entry.extrinsic_params(iterated_num, :) = [];
            data_cluster_entry.intrinsic_params(iterated_num, :) = [];
            data_cluster_entry.affine_matrix(iterated_num, :) = [];
            data_cluster_entry.adjust_matrix(iterated_num, :) = [];
            num_array = [num_array(1 : iterated_num - 1) num_array(iterated_num + 1 : end) 0];
            
            iterated_num = iterated_num - 1;
            data_cluster_entry.num = data_cluster_entry.num - 1;
        else
            data_cluster_entry.linear_ind{iterated_num} = data_cluster_entry.linear_ind{iterated_num}(sampled_ind,:);
            data_cluster_entry.pts_old{iterated_num} = data_cluster_entry.pts_old{iterated_num}(sampled_ind,:);
            data_cluster_entry.pts_new{iterated_num} = data_cluster_entry.pts_new{iterated_num}(sampled_ind,:);
        end
        % data_cluster_entry.linear_ind{iterated_num} = data_cluster_entry.linear_ind{iterated_num}(sampled_ind,:);
        % data_cluster_entry.pts_old{iterated_num} = data_cluster_entry.pts_old{iterated_num}(sampled_ind,:);
        % data_cluster_entry.pts_new{iterated_num} = data_cluster_entry.pts_new{iterated_num}(sampled_ind,:);
        iterated_num = iterated_num + 1;
    end
end
function [data_cluster_entry, data_cluster_prop] = merge_obj_into_data_cluster_entry(data_cluster_entry, org_data_entry, data_cluster_prop, ind)
    cur_ind = data_cluster_entry.num + 1;
    data_cluster_entry.num = cur_ind;
    data_cluster_entry.frame(cur_ind) = org_data_entry.frame;
    data_cluster_entry.linear_ind{cur_ind} = org_data_entry.linear_ind;
    data_cluster_entry.extrinsic_params(cur_ind, :) = org_data_entry.extrinsic_params(:);
    data_cluster_entry.intrinsic_params(cur_ind, :) = org_data_entry.intrinsic_params(:);
    data_cluster_entry.affine_matrix(cur_ind, :) = org_data_entry.affine_matrx(:);
    data_cluster_entry.adjust_matrix(cur_ind, :) = org_data_entry.adjust_matrix(:);
    data_cluster_entry.pts_old{cur_ind} = org_data_entry.pts_old;
    data_cluster_entry.pts_new{cur_ind} = org_data_entry.pts_new;
    
    data_cluster_prop.num_record(ind, cur_ind) = length(org_data_entry.linear_ind);
end
function [data_cluster, data_cluster_prop] = merge_obj_into_data_cluster(obj, data_cluster, data_cluster_prop)
    obj = down_sample_obj(obj);
    for i = 1 : length(obj)
        ind = find_instanceid_pos(data_cluster, obj{i}.instanceId, data_cluster_prop);
        if ind == 0 || ind == -1
            [data_cluster{data_cluster_prop.num + 1}, data_cluster_prop] = init_one_entry(obj{i}, data_cluster_prop);
        else
            [data_cluster{ind}, data_cluster_prop] = merge_obj_into_data_cluster_entry(data_cluster{ind}, obj{i}, data_cluster_prop, ind);
        end
    end
    % [data_cluster, data_cluster_prop] = down_sample_data_cluster(data_cluster, data_cluster_prop);
    [data_cluster, data_cluster_prop] = delete_dull_entry(data_cluster, data_cluster_prop, obj{1}.frame);
end
function obj = down_sample_obj(obj)
    down_sample_rate = 0.1;
    
end
function [data_cluster, data_cluster_prop] = delete_dull_entry(data_cluster, data_cluster_prop, frame)
    it_num = 1;
    while it_num <= data_cluster_prop.num
        if data_cluster{it_num}.frame(data_cluster{it_num}.num) ~= frame
            data_cluster(it_num) = []; data_cluster_prop.num = data_cluster_prop.num - 1; data_cluster_prop.num_record(it_num,:) = [];
            it_num = it_num - 1;
        end
        it_num = it_num + 1;
    end
end
function [inited_entry, data_cluster_prop] = init_one_entry(old_entry, data_cluster_prop)
    inited_entry.num = 1;
    inited_entry.instanceId = old_entry.instanceId;
    inited_entry.color = old_entry.color;
    
    inited_entry.frame = zeros(20, 1);
    inited_entry.extrinsic_params = zeros(20, 16);
    inited_entry.intrinsic_params = zeros(20, 16);
    inited_entry.affine_matrix = zeros(20, 16);
    inited_entry.adjust_matrix = zeros(20, 16);
    inited_entry.linear_ind = cell(20, 1);
    inited_entry.pts_old = cell(20, 1);
    inited_entry.pts_new = cell(20, 1);
    
    inited_entry.frame(1) = old_entry.frame;
    inited_entry.extrinsic_params(1,:) = old_entry.extrinsic_params(:);
    inited_entry.intrinsic_params(1,:) = old_entry.intrinsic_params(:);
    inited_entry.affine_matrix(1,:) = old_entry.affine_matrx(:);
    inited_entry.adjust_matrix(1,:) = old_entry.adjust_matrix(:);
    inited_entry.linear_ind{1} = old_entry.linear_ind;
    inited_entry.pts_old{1} = old_entry.pts_old;
    inited_entry.pts_new{1} = old_entry.pts_new;
    
    
    data_cluster_prop.num_record(data_cluster_prop.num + 1, inited_entry.num) = length(old_entry.linear_ind); 
    data_cluster_prop.num = data_cluster_prop.num + 1;
end
function ind = find_instanceid_pos(data_cluster, instanceId, data_cluster_prop)
    ind = -1;
    if sum(data_cluster_prop.num_record(:)) == 0
        ind = 0;
    else
        for i = 1 : data_cluster_prop.num
            if instanceId == data_cluster{i}.instanceId
                ind = i;
                break;
            end
        end
        
    end
end
function [extrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame)
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, ~, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
end
function path = make_dir()
    father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/visualize_error_boosting_and_initial_guess_results/';
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path = [father_folder DateString '_organized_data'];
    mkdir(path);
end