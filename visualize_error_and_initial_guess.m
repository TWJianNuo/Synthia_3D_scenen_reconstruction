% visualize data
[base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
intrinsic_params = get_intrinsic_matrix();
n = 294; exp_re_path = make_dir();
for frame = 58 : 58
    [data_cluster, data_cluster_prop] = read_organized_data(frame);
    visualize_data_cluster_and_save(data_cluster, data_cluster_prop, exp_re_path, frame)
end
function visualize_data_cluster_and_save(data_cluster, data_cluster_prop, exp_re_path, frame)
    % figure(1); clf;
    % f = figure('visible', 'off'); clf;
    % for i = 1 : data_cluster_prop.num
        % [aligned_pts, ~] = get_transferred_points(data_cluster{i}, data_cluster_prop.num_record(i, :));
        % [aligned_pts, ~] = transfer_and_visualize(data_cluster{i},  data_cluster_prop.num_record(i, :));
        % color = data_cluster{i}.color;
        % scatter3(aligned_pts(:,1),aligned_pts(:,2),aligned_pts(:,3),3,color,'fill'); hold on
    % end
    X = align_all_pts_to_last_frame_and_generate_image(data_cluster);
    imwrite(X, [exp_re_path '/' num2str(frame,'%06d') '.png']);
end
function X = align_all_pts_to_last_frame_and_generate_image(data_cluster)
    figure(1); clf;
    % f = figure('visible', 'off'); clf;
    for i = 1 : length(data_cluster)
        cur_entry = data_cluster{i}; num = max(cur_entry.frame) - min(cur_entry.frame(cur_entry.frame ~= 0)) + 1;
        tot_pt_num = 0; edited_start_point = num - 1; % max_frame = 5;
        if edited_start_point == 0
            edited_start_point = 1;
        end
        % if num > max_frame
        %    edited_start_point = num - max_frame;
        % end
        for j = edited_start_point : num
            tot_pt_num = tot_pt_num + length(cur_entry.linear_ind{j});
        end
        tot_pt = zeros(tot_pt_num, 4); iterated_pt = 1; affine_matrix = reshape(cur_entry.affine_matrix(num, :), [4 4]);
        for q = edited_start_point : num
            adjust_matrix = eye(4);
            for j = num : -1 : q + 1
                cur_adjust_matrix = cur_entry.adjust_matrix(j,:);
                if sum(cur_adjust_matrix) == 0
                    cur_adjust_matrix = eye(4); cur_adjust_matrix = cur_adjust_matrix(:);
                end
                adjust_matrix = reshape(cur_adjust_matrix, [4 4]) * adjust_matrix;
            end
            tot_pt(iterated_pt : iterated_pt + length(cur_entry.linear_ind{q}) - 1, :) = (affine_matrix * inv(adjust_matrix) * (cur_entry.pts_old{q})')';
            iterated_pt = iterated_pt + length(cur_entry.linear_ind{q});
        end
        color = cur_entry.color;
        scatter3(tot_pt(:,1),tot_pt(:,2),tot_pt(:,3),3,color,'fill'); hold on;
    end
    axis equal;
    F = getframe(f); [X, ~] = frame2im(F);
end
function [aligned_pts, frame_belongs] = get_transferred_points(data_cluster_entry, data_cluster_prop_entry)
    tot_num = sum(data_cluster_prop_entry(end-1 : end)); aligned_pts = zeros(tot_num, 4); frame_belongs = zeros(tot_num, 1);
    processed_num = 1; adjuste_matrix = eye(4);
    for i = data_cluster_entry.num : -1 : data_cluster_entry.num - 1
        if sum(data_cluster_entry.adjust_matrix(i,:)) ~= 0
            adjuste_matrix = reshape(data_cluster_entry.adjust_matrix(i,:), [4, 4]) * adjuste_matrix;
        end
        affine_matrix = reshape(data_cluster_entry.affine_matrix(i,:), [4, 4]);
        aligned_pts(processed_num : processed_num + data_cluster_prop_entry(i) - 1, :) = (affine_matrix * adjuste_matrix * (data_cluster_entry.pts_old{i})')';
        frame_belongs(processed_num : processed_num + data_cluster_prop_entry(i) - 1, :) = data_cluster_entry.frame(i);
        processed_num = processed_num + data_cluster_prop_entry(i);
    end
end
function [aligned_pts, frame_belongs] = transfer_and_visualize(data_cluster_entry, data_cluster_prop_entry)
    tot_num = sum(data_cluster_prop_entry(end-1 : end)); aligned_pts = zeros(tot_num, 4); frame_belongs = zeros(tot_num, 1);
    processed_num = 1; adjuste_matrix = eye(4);
    for i = data_cluster_entry.num : -1 : data_cluster_entry.num - 1
        if sum(data_cluster_entry.adjust_matrix(i,:)) ~= 0
            adjuste_matrix = reshape(data_cluster_entry.adjust_matrix(i,:), [4, 4]) * adjuste_matrix;
        end
        affine_matrix = reshape(data_cluster_entry.affine_matrix(i,:), [4, 4]);
        aligned_pts(processed_num : processed_num + data_cluster_prop_entry(i) - 1, :) = (affine_matrix * adjuste_matrix * (data_cluster_entry.pts_old{i})')';
        frame_belongs(processed_num : processed_num + data_cluster_prop_entry(i) - 1, :) = data_cluster_entry.frame(i);
        processed_num = processed_num + data_cluster_prop_entry(i);
    end
end
function [data_cluster, data_cluster_prop] = read_organized_data(frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/visualize_error_boosting_and_initial_guess_results/20_Sep_2018_22_organized_data/';
    loaded_data = load([path num2str(frame, '%06d') '.mat']);
    data_cluster = loaded_data.data_cluster;
    data_cluster_prop = loaded_data.data_cluster_prop;
end
function path = make_dir()
    father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/visualize_error_boosting_and_initial_guess_results/';
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path = [father_folder DateString 'error_boosting_visualization'];
    mkdir(path);
end