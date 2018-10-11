% visualize data
sv_path = make_dir();
[base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
% intrinsic_params = get_intrinsic_matrix();
n = 294; % exp_re_path = make_dir();
exp_severiety(sv_path)
%{
for frame = 58 : 58
    [extrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame);
    [data_cluster, data_cluster_prop] = read_organized_data(frame);
    visualize_data_cluster_and_save(data_cluster, data_cluster_prop, exp_re_path, frame)
end
%}
function exp_severiety(sv_path)
    start_frame_set = [45]; span = 20;
    %{
    for i = 1 : length(start_frame_set)
        for j = 1 : span
            img = show_reconstruction_re(start_frame_set(i), start_frame_set(i) + j);
            save_img(img, sv_path, start_frame_set(i), start_frame_set(i) + j);
        end
    end
    %}
    for i = 1 : length(start_frame_set)
        for j = 1 : span
            s_ind = start_frame_set(i); e_ind = s_ind + j;
            img = show_reconstruction_re(s_ind, e_ind);
            save_img(img, sv_path, s_ind, e_ind);
        end
    end
end
function save_img(img, sv_path, i, j)
    imwrite(img, [sv_path '/' num2str(i) '_' num2str(j) '.png']);
end
function path = make_dir()
    father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/visualize_error_boosting_and_initial_guess_results/';
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path = [father_folder DateString];
    mkdir(path);
end
function img = show_reconstruction_re(start_frame, end_frame)
    f = figure('visible', 'off'); clf;
    frame_tune_matrix = calculate_transformation_matrix(start_frame, end_frame);
    s_data = read_in_org_entry(start_frame); e_data = read_in_org_entry(end_frame); % e_data = align_data_point(s_data, e_data, frame_tune_matrix, start_frame, end_frame);
    % A = align_data_point(s_data, e_data, frame_tune_matrix, start_frame, end_frame)
    % visualize_one_frame_data(s_data, 1); visualize_one_frame_data(e_data, 2, frame_tune_matrix);
    % F = getframe(f); [img] = frame2im(F);
    img = draw_scene(s_data, e_data, frame_tune_matrix);
end
function img = draw_scene(s_data, e_data, frame_tune_matrix)
    affine = s_data{1}.affine_matrx;
    s_all = acquire_all_pts_from_mark(s_data);
    e_all = acquire_all_pts_from_mark(e_data); e_all = (frame_tune_matrix * e_all')';
    s_all = (affine * s_all')'; e_all = (affine * e_all')';
    f = figure('visible', 'off'); clf; scatter3(s_all(:,1),s_all(:,2),s_all(:,3),3,'r','fill'); hold on; scatter3(e_all(:,1),e_all(:,2),e_all(:,3),3,'g','fill');
    F = getframe(f); [img] = frame2im(F);
    
end
function old_pts = acquire_all_pts(num_frame)
    [extrinsic_params, depth, label, instance, rgb] = grab_provided_data(num_frame);
    intrinsic_params = get_intrinsic_matrix();
    old_pts = calculate_old_pts(label, depth, extrinsic_params, intrinsic_params);
end
function old_pts = calculate_old_pts(label, depth_map, extrinsic_params, intrinsic_params)
    building_type = 2; [ix, iy] = find(label == building_type); linear_ind_record = sub2ind(size(depth_map), ix, iy);
    old_pts = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, linear_ind_record);
end
function reconstructed_3d = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
    height = size(depth_map, 1);
    width = size(depth_map, 2);
    x = 1 : height; y = 1 : width;
    [X, Y] = meshgrid(y, x);
    pts = [Y(:) X(:)];
    projects_pts = [pts(valuable_ind,2) .* depth_map(valuable_ind), pts(valuable_ind,1) .* depth_map(valuable_ind), depth_map(valuable_ind), ones(length(valuable_ind), 1)];
    reconstructed_3d = (inv(intrinsic_params * extrinsic_params) * projects_pts')';
end
function pts_all = acquire_all_pts_from_mark(mark)
    pts_all = zeros(0);
    for i = 1 : length(mark)
        pts_all = [pts_all; mark{i}.pts_old];
    end
end
function A = align_data_point(s_data, e_data, frame_tune_matrix, start_frame, end_frame)
    load('adjust_matrix.mat'); A = reshape(param_record(end_frame-1,:), [4,4]); 
    pts_prev_mark1 = acquire_all_pts_from_mark(s_data); pts_prev_mark2 = acquire_all_pts_from_mark(e_data);
    old_pts1 = acquire_all_pts(start_frame); old_pts2 = acquire_all_pts(end_frame); new_old_pts2 = (A * old_pts2')'; new_pts_prev_mark2 = (A * pts_prev_mark2')';
    figure(1); clf; scatter3(old_pts1(:,1), old_pts1(:,2), old_pts1(:,3), 3, 'r', 'fill'); hold on; scatter3(pts_prev_mark1(:,1), pts_prev_mark1(:,2), pts_prev_mark1(:,3), 3, 'b', 'fill'); axis equal; F = getframe(gcf); [X1, Map] = frame2im(F);
    figure(2); clf; scatter3(old_pts2(:,1), old_pts2(:,2), old_pts2(:,3), 3, 'g', 'fill'); hold on; scatter3(pts_prev_mark2(:,1), pts_prev_mark2(:,2), pts_prev_mark2(:,3), 3, 'b', 'fill'); axis equal; F = getframe(gcf); [X2, Map] = frame2im(F);
    figure(3); clf; scatter3(old_pts1(:,1), old_pts1(:,2), old_pts1(:,3), 3, 'r', 'fill'); hold on; scatter3(new_old_pts2(:,1), new_old_pts2(:,2), new_old_pts2(:,3), 3, 'g', 'fill'); axis equal; F = getframe(gcf); [X3, Map] = frame2im(F);
    figure(4); clf; scatter3(pts_prev_mark1(:,1), pts_prev_mark1(:,2), pts_prev_mark1(:,3), 3, 'r', 'fill'); hold on; scatter3(new_pts_prev_mark2(:,1), new_pts_prev_mark2(:,2), new_pts_prev_mark2(:,3), 3, 'g', 'fill'); F = getframe(gcf); [X4, Map] = frame2im(F);
end
function visualize_one_frame_data(org_frame_data, flag, frame_tune_matrix)
    if nargin == 2
        frame_tune_matrix = eye(4);
    end
    for i = 1 : length(org_frame_data)
        cur_data_cluster = org_frame_data{i};
        pts_old = (frame_tune_matrix * (cur_data_cluster.pts_old)')';
        if flag == 1
            scatter3(pts_old(:,1),pts_old(:,2),pts_old(:,3),3,'r','fill'); hold on;
        end
        if flag == 2
            scatter3(pts_old(:,1),pts_old(:,2),pts_old(:,3),3,'g','fill'); hold on;
        end
    end
end
function frame_tune_matrix = calculate_transformation_matrix(start_frame, end_frame)
    load('adjust_matrix.mat')
    frame_tune_matrix = eye(4);
    for i = start_frame : end_frame - 1
        frame_tune_matrix = frame_tune_matrix * reshape(param_record(i, :), [4 4]);
    end
end
function org_entry = read_in_org_entry(frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map/';
    ind = num2str(frame, '%06d');
    loaded = load([path ind '.mat']); org_entry = loaded.prev_mark;
end
function [extrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame)
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, ~, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
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