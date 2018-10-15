clear; clc;
max_frame = 294; frame = 1; s_frame = 50; e_frame = 60; instance_id = 1;
make_dir();
% create_rgb_cluster()
do_multiple_frame_based_optimization();
% check_correctness()
% generate_depth_cluster()
% organize_data_structure_for_multiple_frames();
% org_entry = read_in_org_entry(50); union_single_entry(org_entry)
% [organized_data, depth_map_collection, rgb_collection] = do_preperation(s_frame, e_frame, instance_id);
% cuboid_record = estimate_cubic_shape(organized_data, depth_map_collection);
% visualize_and_save_data(cuboid_record, rgb_collection, organized_data);
% make_dir()
% rectangulars = get_init_guess(mark); optimize_rectangles(rectangulars, depth);
% rectangulars = stack_sampled_pts(rectangulars);
% draw_point_map(rectangulars);
% rectangulars = get_init_guess(mark); rgb = render_image(rgb, rectangulars); 
% draw_point_map(rectangulars); optimize_cubic(rectangulars{1});
% sampled_pts = sample_cubic_by_num(cuboid, sample_pt_num, sample_pt_num); image_size = size(depth_map);
% [pts_estimated_2d, pts_estimated_vlaid, ~, depth] = projectPoints(sampled_pts, intrinsic_params(1:3, 1:3), extrinsic_params, [0,0,0,0,0], [image_size(1) image_size(2)], false);
% sampled_pts = sampled_pts(pts_estimated_vlaid, :); pts_estimated_2d = pts_estimated_2d(pts_estimated_vlaid, :); depth = depth(pts_estimated_vlaid);
% camera_origin = (-extrinsic_params(1:3, 1:3)' * extrinsic_params(1:3, 4))';
% cubics = {cuboid}; [visible_pt_3d, ~, ~] = find_visible_pt_global(cubics, pts_estimated_2d, sampled_pts, depth, intrinsic_params, extrinsic_params, camera_origin);
% fin_params = analytical_gradient_v2(obj.cur_cuboid, intrinsic_params, extrinsic_params, visible_pt_3d, obj.depth_map, obj.new_pts);
% objs{1}.guess(1:5) = fin_params; cx = objs{1}.guess(1); cy = objs{1}.guess(2); theta = objs{1}.guess(3); l = objs{1}.guess(4); w = objs{1}.guess(5); h = objs{1}.guess(6);
% objs{1}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
function do_multiple_frame_based_optimization()
    global path_mul
    max_frame = 294; cubic_cluster = zeros(0);
    for frame = 1 : max_frame
        % rgb = grab_rgb_data(frame);
        save([path_mul num2str(frame) '.mat'])
        rgb = grab_rgb_by_mat(frame);
        [data_cluster, depth_cluster] = read_in_clusters(frame);
        [data_cluster, cubic_cluster] = optimize_cubic_shape_for_data_cluster(data_cluster, depth_cluster, cubic_cluster, frame);
        metric_record = calculate_metric(cubic_cluster, data_cluster, depth_cluster);
        rgb = render_image_building(rgb, cubic_cluster, data_cluster);
        save_results(metric_record, rgb, frame);
        % draw_and_check_r1esults(data_cluster, cubic_cluster, frame)
    end
end
function rgb = grab_rgb_by_mat(frame)
    global computer_flag
    if computer_flag == 1
        local_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/rgb_data/';
    else
        local_path = '/Volumes/Untitled/21_Sep_2018_07_segmentation/rgb_data/';
    end
    load([local_path num2str(frame) '.mat']);
end
function save_results(metric_record, rgb, frame)
    global path_mul
    imwrite(rgb, [path_mul 'rgb_image/' num2str(frame), '.png']);
    if frame == 1
        f1 = fopen([path_mul 'rgb_image/' 'metric.txt'],'w');
    else
        f1 = fopen([path_mul 'rgb_image/' 'metric.txt'],'a');
    end
    print_matrix(f1, [frame]);
    print_matrix(f1, metric_record);
end
function rgb = render_image_building(rgb, cubic_cluster, data_cluster)
    for i = 1 : length(data_cluster)
        intrinsic_params = data_cluster{i}{end}.intrinsic_params;
        extrinsic_params = data_cluster{i}{end}.extrinsic_params * inv(data_cluster{i}{1}.affine_matrx);
        rgb = cubic_lines_of_2d(rgb, cubic_cluster{i}.cuboid, intrinsic_params, extrinsic_params);
    end
end
function create_rgb_cluster()
    max_frame = 294;
    for frame = 1 : max_frame
        rgb = grab_rgb_data(frame);
        save(['/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/rgb_data/' num2str(frame) '.mat'], 'rgb')
    end
end
function [data_cluster, cubic_cluster] = optimize_cubic_shape_for_data_cluster(data_cluster, depth_cluster, cubic_cluster, frame)
    cubic_cluster = get_cubic_cluster(data_cluster, cubic_cluster);
    cubic_cluster = restore_changed_cubics(cubic_cluster, data_cluster);
    [data_cluster, cubic_cluster] = trim_incontinuous_frame(data_cluster, cubic_cluster);
    cubic_cluster = multiple_frame_based_optimization(data_cluster, cubic_cluster, depth_cluster, frame);
end
function cubic_cluster = multiple_frame_based_optimization(data_cluster, cubic_cluster, depth_cluster, frame)
    for i = 1 : length(data_cluster)
        cubic_cluster{i} = optimize_for_single_obj_set(cubic_cluster{i}, data_cluster{i}, depth_cluster, frame, i);
    end
end 
function cubic_record_entry = optimize_for_single_obj_set(cubic_record_entry, objs, depth_cluster, frame_num, obj_ind)
    global path_mul
    save([path_mul num2str(frame_num) '_' num2str(obj_ind) '.mat'])
    % load('/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/15_Oct_2018_09_mul/21_11.mat');
    activation_label = cubic_record_entry.activation_label; depth_cluster = image_blur(depth_cluster);
    sz_depth_map = size(depth_cluster.depth_maps{1}); it_num = 200; loss_record = zeros(it_num, 1); cuboid_record = cell(it_num, 1);
    delta_record_norm = zeros(it_num, 1);
    for i = 1 : it_num
        cuboid = cubic_record_entry.cuboid; visible_pts = cubic_record_entry.visible_pts;
        is_visible_record = get_visible_objs(cuboid, objs); t_objs = objs(is_visible_record);
        if i == 1
            counts_set = get_useable_sample_for_obj_cluster(objs, depth_cluster, visible_pts);
        end
        if ~isempty(t_objs)
            diff_sum = 0; hess_sum = 0; loss_sum = 0;
            t_counts_set = counts_set(is_visible_record, :); delta_norm_record = zeros(length(t_objs),1);
            if sum(sum(t_counts_set)) == 0
                break;
            end
            for j = 1 : length(t_objs)
                if sum(t_counts_set(j,:)) == 0
                    continue;
                end
                depth_map = grab_depth_map(depth_cluster, t_objs{j}); 
                [diff, hess, loss] = accum_for_one_obj(cuboid, t_objs{j}, depth_map, visible_pts(t_counts_set(j,:), :), activation_label);
                % visualize_inner_sit(cuboid, t_objs{j}, depth_map, visible_pts, activation_label, t_counts_set(j,:))
                diff_sum = diff_sum + diff; hess_sum = hess_sum + hess; loss_sum = loss_sum + loss;
                delta_norm_record(j) = norm(get_delta_from_diff_and_hess(diff, hess));
            end
            delta_theta = get_delta_from_diff_and_hess(diff_sum, hess_sum); loss_record(i) = loss_sum; cuboid_record{i} = cubic_record_entry; delta_record_norm(i) = norm(delta_theta);
            % save_visualize(cubic_record_entry, objs, i, frame_num, obj_ind);
            cubic_record_entry = update_cuboid_entry(cubic_record_entry, delta_theta, activation_label, t_objs{1});
            if judge_stop(delta_theta, cubic_record_entry.cuboid, loss_record, delta_record_norm)
                % break;
            end
        else
            break
        end
    end
    save_visualize(cubic_record_entry, t_objs, it_num, frame_num, obj_ind);
    if ~isempty(t_objs)
        save_stem(loss_record, frame_num, obj_ind);
        % figure(1); clf; stem(loss_record,'filled')
    end
end
function depth_map_cluster = image_blur(depth_map_cluster)
    for i = 1 : length(depth_map_cluster)
        depth_map_cluster.depth_maps{i} = imgaussfilt(depth_map_cluster.depth_maps{i},'FilterSize',3);
    end
end
function metric_record = calculate_metric(cuboid_cluster, objs_cluster, depth_map_cluster)
    metric_record = zeros(length(cuboid_cluster), 2);
    for i = 1 : length(cuboid_cluster)
        metric_record(i,1) = metric1_single(cuboid_cluster{i}.cuboid, objs_cluster{i}, depth_map_cluster);
        metric_record(i,2) = metric2_single(cuboid_cluster{i}.cuboid, objs_cluster{i}, depth_map_cluster);
    end
end
function pts_3d = get_all_3d_pts(obj, depth_map, affine)
    linear_ind = obj.linear_ind; extrinsic = obj.extrinsic_params * inv(affine); intrinsic = obj.intrinsic_params;
    depth_val = depth_map(linear_ind); sz_depth = size(depth_map);
    [yy, xx] = ind2sub([sz_depth(1) sz_depth(2)], linear_ind);
    pts_transed = [xx .* depth_val, yy .* depth_val, depth_val, ones(size(depth_val,1),1)];
    pts_3d = (inv(intrinsic * extrinsic) * pts_transed')';
end
function ave_dist = metric2_single(cuboid, objs, depth_map_cluster)
    tot_dist = 0; tot_num = 0;
    for i = 1 : length(objs)
        depth_map = distill_depth_map_frame_depth_cluster(objs{i}, depth_map_cluster);
        pts_3d = get_all_3d_pts(objs{i}, depth_map, objs{1}.affine_matrx);
        [dist, num] = get_dist_and_num_record(cuboid, pts_3d);
        tot_dist = tot_dist + dist; tot_num = tot_num + num;
    end
    ave_dist = tot_dist / tot_num;
end
function [tot_dist, num] = get_dist_and_num_record(cuboid, pts_3d)
    [~, dist_record] = calculate_ave_distance(cuboid, pts_3d);
    num = length(dist_record); tot_dist = sum(dist_record);
end
function metric_single = metric1_single(cuboid, objs, depth_map_cluster)
    sample_num = 10;
    sampled_pts = sample_cubic_by_num(cuboid, sample_num, sample_num);
    diff_pos_sum = 0; diff_inv_sum = 0; num_pt_pos_sum = 0; num_pt_inv_sum = 0;
    for i = 1 : length(objs)
        visible_pts = find_visible_pts(cuboid, sampled_pts, objs{i}.extrinsic_params, objs{i}.intrinsic_params, objs{1}.affine_matrx);
        depth_map = distill_depth_map_frame_depth_cluster(objs{i}, depth_map_cluster);
        [diff_pos, num_pt_pos] = metric_pos(depth_map, visible_pts, objs{i}.extrinsic_params, objs{i}.intrinsic_params, objs{1}.affine_matrx, objs{i}.linear_ind);
        [diff_inv, num_pt_inv] = metric_inv(get_linear_ind(objs{i}, num_pt_pos), depth_map, cuboid, objs{i}.extrinsic_params, objs{i}.intrinsic_params, objs{1}.affine_matrx);
        diff_pos_sum = diff_pos_sum + diff_pos; diff_inv_sum = diff_inv_sum + diff_inv;
        num_pt_pos_sum = num_pt_pos_sum + num_pt_pos; num_pt_inv_sum = num_pt_inv_sum + num_pt_inv;
    end
    metric_single = (diff_pos_sum + diff_inv_sum) / (num_pt_pos_sum + num_pt_inv_sum);
end
function [diff_inv, num] = metric_inv(linear_ind, depth_map, cuboid, extrinsic, intrinsic, affine)
    sz_depth = size(depth_map); extrinsic = extrinsic * inv(affine); sky_val = max(max(depth_map)); diff_inv = 0;
    [yy, xx] = ind2sub([sz_depth(1) sz_depth(2)], linear_ind);     
    [visible_pts_2d, visible_pts_2d_depth, pts_3d_record, visible_pt_label] = find_visible_2d_pts({cuboid}, [xx yy], intrinsic, extrinsic);
    if sum(visible_pt_label) == 0
        diff_inv = 0; num = 0;
    else
        diff_inv = diff_inv + sum((depth_map(linear_ind(visible_pt_label)) - visible_pts_2d_depth).^2); diff_inv = diff_inv + sum((depth_map(linear_ind(~visible_pt_label)) - sky_val).^2);
        num = size(linear_ind, 1);
    end
end
function linear_ind = get_linear_ind(obj, num)
    linear_ind = down_sample_linear_ind(obj.linear_ind, num);
end
function [diff, num_pt] = metric_pos(depth_map, visible_pts, extrinsic, intrinsic, affine, valid_range)
    sz_depth = size(depth_map);
    [pts2d, depth] = project_point_2d(extrinsic, intrinsic, visible_pts(:, 1:3), affine);
    pts2d = round(pts2d); 
    selector = (pts2d(:,1) > 0) & (pts2d(:,1) <= sz_depth(2)) & (pts2d(:,2) > 0) & (pts2d(:,2) < sz_depth(1));
    pts2d = pts2d(selector, :); linear_ind = sub2ind(sz_depth, pts2d(:,2), pts2d(:,1)); ica = ismember(linear_ind, valid_range);
    diff = sum((depth_map(linear_ind(ica)) - depth(ica)).^2); num_pt = size(depth(ica), 1);
    
end
function visible_plane_ind = find_visible_plane(cuboid, intrinsic_params, extrinsic_params)
    params = generate_cubic_params(cuboid); visible_plane_ind = 1 : 4;
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    c1 = [
        xc + 1 / 2 * w * sin(theta);
        yc - 1 / 2 * w * cos(theta);
        h/2;
        1;
        ];
    c2 = [
        xc + 1/2 * l * cos(theta);
        yc + 1/2 * l * sin(theta);
        1/2 * h;
        1
        ];
    c3 = [
        xc - 1/2 * w * sin(theta);
        yc + 1/2 * w * cos(theta);
        1/2 * h;
        1
        ];
    c4 = [
        xc - 1/2 * l * cos(theta);
        yc - 1/2 * l * sin(theta);
        1/2 * h;
        1
        ];
    visible_pt_label = find_visible_pt_global({cuboid}, [c1';c2';c3';c4'], intrinsic_params, extrinsic_params);
    visible_plane_ind = visible_plane_ind(visible_pt_label);
end
function visible_pts = find_visible_pts(cuboid, sampled_pts, extrinsic, intrinsic, affine)
    extrinsic = extrinsic * inv(affine);
    visible_plane_ind = find_visible_plane(cuboid, intrinsic, extrinsic);
    selector = false(size(sampled_pts, 1),1);
    for i = 1 : length(visible_plane_ind)
        selector = selector | (sampled_pts(:, 4) == visible_plane_ind(i));
    end
    visible_pts = sampled_pts(selector, :);
end
function save_stem(loss_record, frame_num, obj_ind)
    global path_mul
    loss_record = loss_record(loss_record~=0);
    figure('visible', 'off'); clf; stem(loss_record', 'filled');
    F = getframe(gcf); [X, ~] = frame2im(F);
    % path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/single_frame_exp/';
    imwrite(X, [path_mul num2str(frame_num) '_' num2str(obj_ind) '_loss' '.png']);
end
function save_visualize(cubic_record_entry, objs, it_num, frame_num, obj_ind)
    global path_mul axis_param_vi
    if mod(it_num, 10) == 0
        figure('visible', 'off'); clf; draw_cubic_shape_frame(cubic_record_entry.cuboid); hold on;
        pts_cubic = cubic_record_entry.visible_pts;
        scatter3(pts_cubic(:,1), pts_cubic(:,2), pts_cubic(:,3), 3, 'r', 'fill'); hold on;
        for i = 1 : length(objs)
            pts = objs{i}.pts_new;
            scatter3(pts(:,1), pts(:,2), pts(:,3), 3, 'g', 'fill'); hold on;
        end
        axis equal; F = getframe(gcf); [X, ~] = frame2im(F);
        %{
        if it_num == 1
            [az,el] = view; limit = axis;
            axis_param_vi = [az,el,limit];
        else
            view(axis_param_vi(1:2));
            axis(axis_param_vi(3:end));
        end
        %}
        % path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/single_frame_exp/';
        imwrite(X, [path_mul num2str(frame_num) '_' num2str(obj_ind) '_' num2str(it_num) '.png']); pause(1)
    end
end
function to_stop = judge_stop(delta, cuboid, diff_record, delta_record_norm)
    params = generate_cubic_params(cuboid); delta_record_norm = delta_record_norm(delta_record_norm~=0); delta_norm_th = 0.00001;
    th = 0.0001; to_stop = false; diff_record = diff_record(diff_record~=0); step_range = 10; th_hold = 0.1;
    if max(abs(delta)) < th || params(4) < 0 || params(5) < 0
        to_stop = true;
    end
    if length(diff_record) > step_range
        if abs(diff_record(end) - diff_record(end - step_range)) < th_hold
            to_stop = true;
        end
    end
    if sum(delta_record_norm < delta_norm_th) > 10
        to_stop = true;
    end
end
function cubic_record = update_cuboid_entry(cubic_record, delta_theta, activation_label, cur_obj)
    cuboid = cubic_record.cuboid; params = generate_cubic_params(cuboid); activation_label = (activation_label == 1);
    %{
    ratio = max(abs(delta_theta) ./ (0.05 * abs(params(activation_label)))); 
    if ratio > 1
        delta_theta = delta_theta / ratio;
    end
    %}
    params(activation_label) = params(activation_label) + delta_theta;
    n_cuboid = generate_center_cuboid_by_params(params); visible_pts = sample_cubic_by_num(n_cuboid, 10, 10);
    % visible_pts = acquire_visible_sampled_points(n_cuboid, cur_obj.intrinsic_params, cur_obj.extrinsic_params, cur_obj.affine_matrx);
    visible_pts = find_visible_pts(cuboid, visible_pts, cur_obj.extrinsic_params, cur_obj.intrinsic_params, cur_obj.affine_matrx);
    cubic_record.cuboid = n_cuboid; cubic_record.visible_pts = visible_pts;
end
function delta = get_delta_from_diff_and_hess(sum_diff, sum_hess)
    if sum(isnan(sum_diff)) + sum(sum(isnan(sum_hess))) > 0
        delta = 0;
    else
        delta = sum_diff * inv(sum_hess);
    end
end
function depth_map = grab_depth_map(depth_cluster, cur_obj)
    frame_ind = cur_obj.frame;
    ind = find(depth_cluster.frame_ind == frame_ind); depth_map = depth_cluster.depth_maps{ind};
end
function visualize_inner_sit(cuboid, cur_obj, depth_map, visible_pts, activation_label, valid_flag)
    visible_pts = visible_pts(valid_flag, :); visible_pt_to_plot = visible_pts(:,1:3);
    
    tot_pos_num = size(visible_pts, 1); linear_ind = cur_obj.linear_ind; 
    pts_3d = get_all_3d_pts(cur_obj, depth_map, cur_obj.affine_matrx);
    linear_ind = down_sample_linear_ind(linear_ind, tot_pos_num);
    visible_pts = [visible_pts(:, 5) visible_pts(:, 6) visible_pts(:, 4)];
    extrinsic_param = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic_param = cur_obj.intrinsic_params;
    color1 = rand(size(linear_ind,1),3); color2 = rand(size(visible_pts,1),3);
    
    % [x_inv_record, dominate_pts, dominate_color] = visualize_combine_multi(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pts, activation_label, color1, color2);
    [x_inv_record, dominate_pts, dominate_color] = visualize_combine_multi(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pts, activation_label);
    [depth_map_projected] = map_pts_to_depth_map(cur_obj.extrinsic_params, cur_obj.intrinsic_params, cur_obj.affine_matrx, dominate_pts, depth_map, dominate_color);
    hold on; scatter3(pts_3d(:,1), pts_3d(:,2), pts_3d(:,3), 4, 'k', 'fill');
    figure(2); clf; imshow(depth_map_projected)
end
function [depth_map, pts_2d] = map_pts_to_depth_map(extrinsic, intrinsic, affine, pts, depth_map, colors)
    [pts_2d, depth_vals] = project_point_2d(extrinsic, intrinsic, pts, affine); pts_2d = round(pts_2d);
    depth_map = mark_on_depth_map(depth_map, pts_2d, colors);
end
function depth_map = mark_on_depth_map(depth_map, pos, color)
    depth_map = uint8(255 * (depth_map / max(max(depth_map))));
    depth_map = cat(3,depth_map,depth_map,depth_map);
    depth_map = insertShape(depth_map,'FilledCircle',[pos(:,1) pos(:,2) ones(length(pos(:,1)),1) * 1],'LineWidth',1,'Color',uint8(255 * color));
end
function depth_map = map_one_obj_to_depth_map(depth_map, cur_obj, cur_affine)
    extrinsic = cur_obj.extrinsic_params; intrinsic = cur_obj.intrinsic_params; pts = cur_obj.pts_new; affine = cur_affine; [height, width] = size(depth_map);
    [pts_2d, depth_vals] = project_point_2d(extrinsic, intrinsic, pts, affine); pts_2d = round(pts_2d); 
    selector = pts_2d(:,1) >= 1 & pts_2d(:,1) <= width & pts_2d(:,2) >=1 & pts_2d(:,2) <= height;
    pts_2d = pts_2d(selector, :); depth_vals = depth_vals(selector);
    linear_ind = sub2ind(size(depth_map), pts_2d(:,2), pts_2d(:,1));
    depth_map(linear_ind) = depth_vals;
end
function [diff, hess, loss] = accum_for_one_obj(cuboid, cur_obj, depth_map, visible_pts, activation_label)
    tot_pos_num = size(visible_pts, 1); linear_ind = cur_obj.linear_ind; tot_inv_num = size(linear_ind, 1);
    linear_ind = down_sample_linear_ind(linear_ind, tot_pos_num);
    visible_pts = [visible_pts(:, 5) visible_pts(:, 6) visible_pts(:, 4)];
    extrinsic_param = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic_param = cur_obj.intrinsic_params;
    [diff, hess, loss] = analytical_gradient_combined_v2_mult_frame(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pts, activation_label);
    % visualize_combine_multi(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pts, activation_label);
end
function new_linear_ind = down_sample_linear_ind(org_linear_ind, new_num)
    if length(org_linear_ind) > new_num
        indices = unique(round(linspace(1, length(org_linear_ind), new_num)));
        new_linear_ind = org_linear_ind(indices);
    end
end
function is_visible_record = get_visible_objs(cuboid, objs)
    is_visible_record = true(size(objs,1),1);
    for i = 1 : length(objs)
        is_visible_record(i) = judge_is_still_visible(cuboid, objs{i});
    end
end
function counts_set = get_useable_sample_for_obj_cluster(objs, depth_cluster, visible_pts)
    counts_set = false(size(objs, 1), size(visible_pts,1)); sz_depth_map = size(depth_cluster.depth_maps{1});
    for i = 1 : length(objs)
        counts_set(i,:) = get_useable_sample_pt(objs{i}, sz_depth_map, visible_pts);
        [depth_map, ~] = distill_depth_map_frame_depth_cluster(objs{i}, depth_cluster);
        counts_set(i,:) = counts_set(i,:) & exclude_border_points(visible_pts, objs{i}, depth_map)';
    end
    %{
    re = ~counts_set(1,:);
    for i = 1 : size(counts_set, 1)
        re = re & (~counts_set(i,:));
    end
    for i = 1 : size(counts_set, 1)
        tmp = counts_set(i,:); tmp(re) = true;
        counts_set(i, :) = tmp;
    end
    %}
end
function selector = exclude_border_points(visible_pts, obj, image)
    grad_record = zeros(size(visible_pts,1),2); grad_th = 100; 
    extrinsic = obj.extrinsic_params; intrinsic = obj.intrinsic_params; affine = obj.affine_matrx;
    location = round(project_point_2d(extrinsic, intrinsic, visible_pts(:,1:3), affine));
    for i = 1 : size(location,1)
        grad_record(i,:) = abs([interpImg(image, [location(i,1) + 1, location(i,2)]) - interpImg(image, [location(i,1), location(i,2)]), interpImg(image, [location(i,1), location(i,2) + 1]) - interpImg(image, [location(i,1), location(i,2)])]);
    end
    selector = (grad_record(:,1) < grad_th) & (grad_record(:,2) < grad_th);
end
function counts = get_useable_sample_pt(cur_obj, sz_depth_map, visible_pts)
    extrinsic = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic = cur_obj.intrinsic_params; height = sz_depth_map(1); width = sz_depth_map(2);
    sampled_pts = visible_pts;
    counts = true(size(sampled_pts,1),1);
    linear_ind = cur_obj.linear_ind; sampled_pts_3d = sampled_pts(:, 1:3); sampled_pts_3d = [sampled_pts_3d ones(size(sampled_pts_3d,1), 1)];
    pts_2d = project_point_2d(extrinsic, intrinsic, sampled_pts_3d); selector = pts_2d(:,1) >= 1 & pts_2d(:,1) <= width & pts_2d(:,2) >=1 & pts_2d(:,2) <= height;
    pts_2d = round(pts_2d(selector, :)); counts(~selector) = false;
    linear_2d = sub2ind([height, width], pts_2d(:,2), pts_2d(:,1)); lia = ismember(linear_2d, linear_ind); counts(counts) = lia;
    
end

function cubic_cluster = restore_changed_cubics(cubic_cluster, data_cluster)
    default_activation_label = [1 1 1 1 1 0];
    for i = 1 : length(data_cluster)
        if ~isequal(data_cluster{i}{1}.restore_matrix, eye(4))
            [cuboid, visible_pts] = restore_new_cubic_shape(cubic_cluster{i}.cuboid, data_cluster{i}{1});
            cubic_cluster{i}.cuboid = cuboid; cubic_cluster{i}.visible_pts = visible_pts;
            cubic_cluster{i}.instanceId = data_cluster{i}{1}.instanceId;
            cubic_cluster{i}.activation_label = default_activation_label;
        end
    end
    for i = 1 : length(data_cluster)
        [cuboid, visible_pts] = re_estimate_cubic_entry(cubic_cluster{i}.cuboid, data_cluster{i});
        cubic_cluster{i}.cuboid = cuboid; cubic_cluster{i}.visible_pts = visible_pts;
        cubic_cluster{i}.instanceId = data_cluster{i}{1}.instanceId;
        cubic_cluster{i}.activation_label = default_activation_label;
    end
end
function [cuboid, sampled_pts] = re_estimate_cubic_entry(old_cuboid, objs)
    % Adjust initial guess for the new coming points
    tot_pt_num = 0; iou_th = 0.2;
    for i = 1 : length(objs)
        tot_pt_num = tot_pt_num + length(objs{i}.linear_ind);
    end
    tot_pts3d = zeros(tot_pt_num, 4); cur_pos = 1;
    for i = 1 : length(objs)
        tot_pts3d(cur_pos : cur_pos + length(objs{i}.linear_ind) - 1, :) = objs{i}.pts_new;
        cur_pos = cur_pos + length(objs{i}.linear_ind);
    end
    [~, new_cuboid] = estimate_rectangular(tot_pts3d);
    iou = calculate_IOU(old_cuboid, new_cuboid);
    if iou < iou_th
        cuboid_fin = new_cuboid;
    else
        cuboid_fin = old_cuboid;
    end
    [cuboid, sampled_pts] = tune_cubic_shape_to_specific_entry(cuboid_fin, objs{end});
end
function [cuboid, sampled_pts] = restore_new_cubic_shape(old_cubic, cur_obj)
    restore_matrix = cur_obj.restore_matrix; affine_matrix = cur_obj.affine_matrx;
    params = generate_cubic_params(old_cubic);
    bottom_pts = old_cubic{5}.pts(1:4,:); bottom_pts(:,3) = 0; bottom_pts = [bottom_pts ones(size(bottom_pts, 1),1)];
    bottom_pts = (affine_matrix * restore_matrix * bottom_pts')'; mean_pts = mean(bottom_pts); cx = mean_pts(1); cy = mean_pts(2); 
    params(2) = cx; params(3) = cy; 
    pts_on_first_plane = old_cubic{1}.pts(1:2, :); pts_on_first_plane = [pts_on_first_plane ones(size(pts_on_first_plane,1),1)]; 
    pts_on_first_plane = (affine_matrix * restore_matrix * pts_on_first_plane')';
    pts_on_first_plane = pts_on_first_plane(:,1:2); dir_vector = pts_on_first_plane(2,:) - pts_on_first_plane(1,:);
    dir_vector = dir_vector / norm(dir_vector); S = dir_vector(2); C = dir_vector(1);
    theta = angleCalc(S, C, 'rad');
    params(1) = theta;
    new_cuboid = generate_center_cuboid_by_params(params);
    [cuboid, sampled_pts] = tune_cubic_shape_to_specific_entry(new_cuboid, cur_obj);
end
function draw_and_check_results(data_cluster, cubic_cluster, frame)   
    figure('visible', 'off'); clf;
    for i = 1 : length(data_cluster)
        cuboid = cubic_cluster{i}.cuboid; vs_pts = cubic_cluster{i}.visible_pts;
        draw_cubic_shape_frame(cuboid); hold on; scatter3(vs_pts(:,1), vs_pts(:,2), vs_pts(:,3), 3, 'g', 'fill'); hold on;
        for j = 1 : length(data_cluster{i})
            vs_pts = data_cluster{i}{j}.pts_new; color = data_cluster{i}{j}.color; hold on; scatter3(vs_pts(:,1), vs_pts(:,2), vs_pts(:,3), 3, color, 'fill');
        end
    end
    axis equal; F = getframe(gcf); [X, ~] = frame2im(F);
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/cubic_check/';
    imwrite(X, [path num2str(frame) '.png']);
end
function [data_cluster, cubic_cluster] = trim_incontinuous_frame(data_cluster, cubic_cluster)
    default_activation_label = [1 1 1 1 1 0];
    for i = 1 : length(data_cluster)
        cuboid = cubic_cluster{i}.cuboid; [cuboid, sampled_pts] = tune_cubic_shape_to_specific_entry(cuboid, data_cluster{i}{end});
        cubic_cluster{i}.cuboid = cuboid; cubic_cluster{i}.visible_pts = sampled_pts;
        is_valid_array = fill_is_valid(cuboid, data_cluster{i}); cubic_cluster{i}.is_valid = is_valid_array;
    end
end
function is_valid_array = fill_is_valid(cuboid, entries)
    is_valid_array = zeros(length(entries), 1);
    for i = 1 : length(entries)
        is_valid_array(i) = judge_is_still_visible(cuboid, entries{i});
    end
end
function [cuboid, sampled_pts] = tune_cubic_shape_to_specific_entry(cuboid, cur_obj)
    extrinsic = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic = cur_obj.intrinsic_params;
    [cuboid, is_valid] = tune_cuboid(cuboid, extrinsic, intrinsic);
    sampled_pts = acquire_visible_sampled_points(cuboid, cur_obj.intrinsic_params, cur_obj.extrinsic_params, cur_obj.affine_matrx);
end
function is_visible = judge_is_still_visible(cubic, cur_obj)
    extrinsic = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic = cur_obj.intrinsic_params;
    is_visible = jude_is_first_and_second_plane_visible(cubic, intrinsic, extrinsic);
end
function new_cubic_cluster = get_cubic_cluster(data_cluster, cubic_cluster)
    new_cubic_cluster = cell(length(data_cluster), 1); default_activation_label = [1 1 1 1 1 0];
    for i = 1 : length(data_cluster)
        pos = find_pos_of_certain_ind_in_cubic_cluster(cubic_cluster, data_cluster{i}{1}.instanceId);
        if pos == 0
            [cuboid, visible_pts, is_one_plane_visible] = estimate_cubic_shape_by_obj(data_cluster{i}{1});
            new_cubic_cluster{i}.cuboid = cuboid; new_cubic_cluster{i}.visible_pts = visible_pts;
            new_cubic_cluster{i}.instanceId = data_cluster{i}{1}.instanceId;
            new_cubic_cluster{i}.activation_label = default_activation_label;
            new_cubic_cluster{i}.is_valid = is_one_plane_visible;
        else
            new_cubic_cluster{i} = cubic_cluster{pos};
        end
    end
end
function [cuboid, sampled_pts, is_valid] = estimate_cubic_shape_by_obj(cur_obj)
    pts_3d = cur_obj.pts_new; extrinsic = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic = cur_obj.intrinsic_params;
    [~, cuboid] = estimate_rectangular(pts_3d);
    [cuboid, is_valid] = tune_cuboid(cuboid, extrinsic, intrinsic);
    sampled_pts = acquire_visible_sampled_points(cuboid, cur_obj.intrinsic_params, cur_obj.extrinsic_params, cur_obj.affine_matrx);
    % figure(1); clf; draw_cubic_shape_frame(cuboid); hold on; scatter3(pts_3d(:,1), pts_3d(:,2), pts_3d(:,3), 3, 'g', 'fill')
    % hold on; scatter3(sampled_pts(:,1),sampled_pts(:,2),sampled_pts(:,3),3,'b','fill')
end
function pos = find_pos_of_certain_ind_in_cubic_cluster(cubic_cluster, instance_id)
    pos = 0;
    for i = 1 : length(cubic_cluster)
        if cubic_cluster{i}.instanceId == instance_id
           pos = i;
        end
    end
end

function rgb = grab_rgb_data(frame)
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
end
function generate_depth_cluster()
    max_frame_num = 294; depth_cluster = init_depth_cluster();
    for i = 1 : max_frame_num
        data_cluster = read_in_data_cluster(i);
        cur_depth_map = read_in_depth_map(i);
        depth_cluster = merge_depth_cluster(cur_depth_map, depth_cluster, i);
        depth_cluster = clean_depth_cluster(data_cluster, depth_cluster);
        save_depth_cluster(i, depth_cluster);
    end
end
function check_correctness()
    max_frame_num = 294;
    for i = 1 : max_frame_num
        [data_cluster, depth_cluster] = read_in_clusters(i);
        new_depth_cluster = map_data_cluster_to_depth_cluster(data_cluster, depth_cluster);
        max_diff = find_differences_between_depth_cluster(depth_cluster, new_depth_cluster);
        record_diff(i, max_diff)
    end
end
function record_diff(frame, max_diff)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_organized/';
    if frame == 1
        f1 = fopen([path '/' 'diff.txt'],'w');
    else
        f1 = fopen([path '/' 'diff.txt'],'a');
    end
    print_matrix(f1, max_diff);
end
function max_diff = find_differences_between_depth_cluster(cluster1, cluster2)
    for i = 1 : length(cluster1.depth_maps)
        depth1 = cluster1.depth_maps{i};
        depth2 = cluster2.depth_maps{i};
        max_diff = max(max(abs(depth1 - depth2)));
    end
end
function depth_cluster = map_data_cluster_to_depth_cluster(data_cluster, depth_cluster)
    for i = 1 : length(data_cluster)
        cur_affine = data_cluster{i}{1}.affine_matrx;
        for j = 1 : length(data_cluster{i})
            cur_obj = data_cluster{i}{j};
            [depth_map, pos_ind] = distill_depth_map_frame_depth_cluster(cur_obj, depth_cluster);
            depth_cluster.depth_maps{pos_ind} = map_one_obj_to_depth_map(depth_map, cur_obj, cur_affine);
        end
    end
end
function [depth_map, pos_ind] = distill_depth_map_frame_depth_cluster(cur_obj, depth_cluster)
    frame_num = cur_obj.frame; pos_ind = 0;
    for i = 1 : length(depth_cluster.frame_ind)
        if frame_num == depth_cluster.frame_ind(i)
            pos_ind = i;
        end
    end
    depth_map = depth_cluster.depth_maps{pos_ind};
end
function [data_cluster, depth_cluster] = read_in_clusters(frame)
    global computer_flag
    if computer_flag == 1
        path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_organized/';
    else
        path = '/Volumes/Untitled/21_Sep_2018_07_segmentation/Instance_map_organized/';
    end
    holder1 = load([path num2str(frame, '%06d') '.mat']); data_cluster = holder1.data_cluster; data_cluster = correct_affine_error(data_cluster);
    holder2 = load([path num2str(frame, '%06d') '_d.mat']); depth_cluster = holder2.depth_cluster;
end
function data_cluster = correct_affine_error(data_cluster)
    for i = 1 : length(data_cluster)
        for j = 2 : length(data_cluster{i})
            data_cluster{i}{j}.affine_matrx = data_cluster{i}{1}.affine_matrx;
        end
    end
end
function save_depth_cluster(frame, depth_cluster)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_organized/';
    save([path, num2str(frame, '%06d') , '_d.mat'], 'depth_cluster');
end
function depth_cluster = clean_depth_cluster(data_cluster, depth_cluster)
    earliest_frame = find_earliest_frame_num(data_cluster); selector = false(length(depth_cluster.frame_ind), 1);
    for i = 1 : length(depth_cluster.frame_ind)
        if depth_cluster.frame_ind(i) < earliest_frame
            selector(i) = true;
        end
    end
    depth_cluster.depth_maps(selector) = []; depth_cluster.frame_ind(selector) = [];
end
function depth_cluster = init_depth_cluster()
    depth_cluster.depth_maps = cell(0);
    depth_cluster.frame_ind = zeros(0);
end
function depth_cluster = merge_depth_cluster(depth_map, depth_cluster, frame)
    depth_cluster.depth_maps{end+1} = depth_map;
    depth_cluster.frame_ind(end+1) = frame;
end
function earliest_frame = find_earliest_frame_num(data_cluster)
    earliest_frame = 100000;
    for i = 1 : length(data_cluster)
        if earliest_frame > data_cluster{i}{1}.frame
            earliest_frame = data_cluster{i}{1}.frame;
        end
    end
end
function depth = read_in_depth_map(frame)
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
end
function map_to_2d(data_cluster, depth_map)
end
function data_cluster = read_in_data_cluster(frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_organized/';
    mounted = load([path, num2str(frame, '%06d') , '.mat']); data_cluster = mounted.data_cluster;
end
function organize_data_structure_for_multiple_frames()
    max_hold_frame_num = 10; max_frame = 294; data_cluster = cell(0);
    for i = 1 : max_frame
        org_entry = read_in_org_entry(i);
        data_cluster = add_data_into_cluster(data_cluster, org_entry, i);
        data_cluster = delete_vanished_entry(data_cluster, i);
        data_cluster = trim_entry_exceeding_max_frame(data_cluster, max_hold_frame_num);
        % X = visualize_data_cluster(data_cluster);
        X = 1;
        save_all(i, X, data_cluster);
    end
end
function save_all(frame, X, data_cluster)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_organized/';
    if frame == 1
        mkdir(path);
    end
    save([path num2str(frame, '%06d') '.mat'], 'data_cluster');
    % imwrite(X, [path num2str(frame, '%06d') '.png']);
end
function X = visualize_data_cluster(data_cluster)
    sample_rate = 10;
    f = figure('visible', 'off'); clf;
    for i = 1 : length(data_cluster)
        for j = 1 : length(data_cluster{i})
            pts = data_cluster{i}{j}.pts_new;
            pts = pts(1:sample_rate:end, :);
            color = data_cluster{i}{j}.color;
            scatter3(pts(:,1), pts(:,2), pts(:,3), 3, color, 'fill'); hold on;
        end
    end
    axis equal; F = getframe(gcf); [X, ~] = frame2im(F);
end
function data_cluster = trim_entry_exceeding_max_frame(data_cluster, max_frame_num)
    for i = 1 : length(data_cluster)
        if length(data_cluster{i}) > max_frame_num
            [data_cluster{i}, restore_matrix] = back_prop_one_frame(data_cluster{i});
            data_cluster{i}(1) = [];
            data_cluster{i}{1}.restore_matrix = restore_matrix;
        else
            idt_mx = eye(4);
            data_cluster{i}{1}.restore_matrix = idt_mx;
        end
    end
end
function [entry, restore_matrix] = back_prop_one_frame(entry)
    load('adjust_matrix.mat'); cur_frame = entry{1}.frame;
    restore_matrix = inv(reshape(param_record(cur_frame, :), [4,4])); 
    for i = 2 : length(entry)
        entry{i}.pts_old = (restore_matrix * entry{i}.pts_old')';
        entry{i}.pts_new = (entry{2}.affine_matrx * entry{i}.pts_old')';
        entry{i}.extrinsic_params = entry{i}.extrinsic_params * reshape(param_record(cur_frame, :), [4,4]);
    end
    restore_matrix = restore_matrix * inv(entry{1}.affine_matrx);
end
function data_cluster = add_data_into_cluster(data_cluster, cur_entry, frame)
    for i = 1 : length(cur_entry)
        obj = cur_entry{i};
        position_ind = find_pos_in_data_cluster(data_cluster, obj);
        if position_ind == 0
            data_cluster = concatinate_data_entry_to_cluster(data_cluster, {obj});
        else
            data_cluster = add_data_entry(data_cluster, obj, position_ind, frame);
        end
    end
end
function data_cluster = delete_vanished_entry(data_cluster, frame_num)
    selector = false(length(data_cluster), 1);
    for i = 1 : length(data_cluster)
        if data_cluster{i}{end}.frame ~= frame_num
            selector(i) = true;
        end
    end
    data_cluster(selector) = [];
end
function data_cluster = add_data_entry(data_cluster, to_add_obj, ind, frame_end)
    frame_first = data_cluster{ind}{1}.frame;
    frame_tune_matrix = calculate_transformation_matrix(frame_first, frame_end);
    to_add_obj = adjust_the_obj(to_add_obj, data_cluster{ind}{1}, frame_tune_matrix);
    data_cluster{ind}{end+1,1} = to_add_obj;
end
function data_cluster = concatinate_data_entry_to_cluster(data_cluster, cur_entry)
    data_cluster{end+1, 1} = cur_entry;
end
function position_ind = find_pos_in_data_cluster(data_cluster, obj)
    position_ind = 0; 
    if isempty(data_cluster)
        position_ind = 0;
    end
    for i = 1 : length(data_cluster)
        if data_cluster{i}{1}.instanceId == obj.instanceId
            position_ind = i;
        end
    end
end
function org_entry = read_in_org_entry(frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_unioned/';
    ind = num2str(frame, '%06d');
    loaded = load([path ind '.mat']); org_entry = loaded.org_entry;
end
function visualize_and_save_data(cuboid_record, rgb_collection, organized_data)
    cuboid_record = cuboid_record(~cellfun('isempty',cuboid_record)) ;
    path = make_dir(); selected_frame = [50]; s_frame = 50;
    for i = 1 : length(cuboid_record)
        cuboid = cuboid_record{i};
        for j = 1 : length(selected_frame)
            ind = selected_frame(j) - s_frame + 1;
            rgb = rgb_collection{ind};
            pts_new = organized_data{ind}.pts_new; 
            intrinsic_params = organized_data{ind}.intrinsic_params;  
            extrinsic_params = organized_data{ind}.extrinsic_params * inv(organized_data{ind}.affine_matrx);
            X = plot_3d(cuboid, pts_new);
            rgbImage = plot2d(rgb,  organized_data{ind}.linear_ind, cuboid); rgbImage = render_img(rgbImage, cuboid, intrinsic_params, extrinsic_params);
            save_img(i, selected_frame(j), X, rgbImage, path);
        end
    end
end
function rgb = render_img(rgb, cuboid, intrinsic_params, extrinsic_params)
    rgb = cubic_lines_of_2d(rgb, cuboid, intrinsic_params, extrinsic_params);
end
function save_img(ind, selected_frame, x, rgb, path)
    imwrite(x, [path num2str(ind) '_' num2str(selected_frame) '_3d.png']);
    imwrite(rgb, [path num2str(ind) '_' num2str(selected_frame) '_2d.png']);
end
function rgbImage = plot2d(rgb, linear_ind, cuboid)
    r = rgb(:,:,1); g = rgb(:,:,2); b = rgb(:,:,3);
    r(linear_ind) = 0; g(linear_ind) = 255; b(linear_ind) = 0;
    rgbImage = cat(3, r, g, b); imshow(rgbImage)
end
function X = plot_3d(cuboid, pts_new)
    f = figure('visible', 'off'); clf; draw_cubic_shape_frame(cuboid); hold on; scatter3(pts_new(:,1), pts_new(:,2), pts_new(:,3), 3, 'g', 'fill');
    F = getframe(f); [X, Map] = frame2im(F);
end
function cuboid_record = estimate_cubic_shape(organized_data, depth_map_collection)
    cuboid_record = cell(length(organized_data),1); num = length(organized_data);
    for i = 1 : num
        if i == 1
            pts = organized_data{1}.pts_new;
            [params, cuboid] = estimate_rectangular(pts);
        end
        sampled_visible_pts_set = find_visible_3d(organized_data(1:i), cuboid);
        % visualize_the_scene(organized_data(1:i), cuboid, sampled_visible_pts_set)
        cuboid = analytical_gradient_multiple_frame(cuboid, depth_map_collection(1:i), sampled_visible_pts_set, organized_data(1:i));
        cuboid_record{i} = cuboid;
    end
end
function visualize_the_scene(organized_data, cuboid, sampled_visible_pts_set)
    for i = 1 : length(sampled_visible_pts_set)
        pts = sampled_visible_pts_set{i}; pts_gt = organized_data{i}.pts_new;
        figure(1); clf; scatter3(pts(:,1), pts(:,2), pts(:,3), 3, 'r', 'fill'); hold on; draw_cubic_shape_frame(cuboid); 
        hold on; scatter3(pts_gt(:,1), pts_gt(:,2), pts_gt(:,3), 3, 'g', 'fill'); axis equal;
    end 
end


function [organized_data, depth_map_collection, rgb_collection] = do_preperation(s_frame, e_frame, instance_id)
    depth_map_collection = cell(e_frame - s_frame + 1, 1); prev_mark_collection = cell(e_frame - s_frame + 1, 1); rgb_collection = cell(e_frame - s_frame + 1, 1);
    for frame = s_frame : e_frame
        ind = frame - s_frame + 1;
        [mark, extrinsic_params, intrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame);
        depth_map_collection{ind} = depth; prev_mark_collection{ind} = read_in_org_entry(frame);
        rgb_collection{ind} = rgb;
    end
    organized_data = organize_data(s_frame, e_frame, instance_id, prev_mark_collection);
    
    % visualize_to_check(organized_data, depth_map_collection);
end
function sampled_visible_pts_set = find_visible_3d(organized_data, cuboid)
    sampled_visible_pts_set = cell(length(organized_data),1);
    for i = 1 : length(organized_data)
        P = organized_data{i}.intrinsic_params; A = organized_data{i}.affine_matrx; T = organized_data{i}.extrinsic_params;
        visible_pts = acquire_visible_sampled_points(cuboid, P, T, A);
        sampled_visible_pts_set{i} = visible_pts;
    end
end
function visualize_to_check(organized_data, depth_map_collection)
    figure(1); clf;
    for i = 1 : length(organized_data)
        pts_new = organized_data{i}.pts_new; color = rand(1,3);
        hold on; scatter3(pts_new(:,1), pts_new(:,2), pts_new(:,3), 3, color, 'fill');
        project_to_depth_map(depth_map_collection{i}, organized_data{i})
    end
end
function project_to_depth_map(depth_map, the_obj)
    org_depth = depth_map;
    extrinsic = the_obj.extrinsic_params * inv(the_obj.affine_matrx); intrinsic = the_obj.intrinsic_params; pts = the_obj.pts_new;
    [pts_2d, depth_val] = project_point_2d(extrinsic, intrinsic, pts); pts_2d = round(pts_2d); linear_ind = sub2ind(size(depth_map), pts_2d(:,2), pts_2d(:,1));
    depth_map(linear_ind) = depth_val; % figure(1); clf; imshow(depth_map / max(max(depth_map)));
end
function frame_tune_matrix = calculate_transformation_matrix(start_frame, end_frame)
    load('adjust_matrix.mat')
    frame_tune_matrix = eye(4);
    for i = start_frame : end_frame - 1
        frame_tune_matrix = frame_tune_matrix * reshape(param_record(i, :), [4 4]);
    end
end
function organized_data = organize_data(s_frame, e_frame, instance_id, prev_mark_collection)
    org_obj = choose_specific_ind(prev_mark_collection{1}, instance_id);
    organized_data = cell(e_frame - s_frame + 1, 1); organized_data{1} = org_obj;
    for i = s_frame + 1 : e_frame
        ind = i - s_frame + 1;
        frame_tune_matrix = calculate_transformation_matrix(s_frame, i);
        the_obj = choose_specific_ind(prev_mark_collection{ind}, instance_id);
        if isempty(the_obj)
            break;
        end
        organized_data{ind} = adjust_the_obj(the_obj, org_obj, frame_tune_matrix);
    end
    organized_data = organized_data(~cellfun('isempty',organized_data)) ;
end
function the_obj = adjust_the_obj(the_obj, org_obj, frame_tune_matrix)
    the_obj.extrinsic_params = the_obj.extrinsic_params * inv(frame_tune_matrix);
    the_obj.pts_old = (frame_tune_matrix * the_obj.pts_old')';
    the_obj.pts_new = (org_obj.affine_matrx * the_obj.pts_old')';
end
function the_obj = choose_specific_ind(mark, ind)
    the_obj = cell(0);
    for i = 1 : length(mark)
        if mark{i}.instanceId == ind
            the_obj = mark{i};
        end
    end
end
function optimize_rectangles(rectangulars, depth_map)
    it_num = 100;
    for i = 1 : it_num
        optimize_cubic_shape(rectangulars{1}, depth_map, i);
    end
end
function [best_params, re_depth_map, space_map, stem_map] = optimize_cubic_shape(rectangular, depth_map, count_num)
    %{
    cuboid = rectangular.cur_cuboid; P = rectangular.intrinsic_params; A = rectangular.affine_matrx; T = rectangular.extrinsic_params;
    cuboid = edit_cubic_shape(cuboid); 
    init_cuboid = cuboid; cuboid = mutate_cuboid(cuboid); [cuboid, is_valide] = tune_cuboid(cuboid, T*inv(A), P);
    rectangular.cur_cuboid = cuboid;
    sampled_visible_pts = acquire_visible_sampled_points(rectangular, P, T, A);
    [processed_depth_map, lin_ind, pts_3d_record] = generate_cubic_depth_map_by(cuboid, P, T*inv(A), depth_map, sampled_visible_pts);
    sampled_visible_pts = [sampled_visible_pts(:, 5:6) sampled_visible_pts(:, 4)];
    %}
    load('trail_inv.mat');
    init_cuboid = cuboid; cuboid = mutate_cuboid(cuboid); [cuboid, is_valide] = tune_cuboid(cuboid, T*inv(A), P);
    [best_params, re_depth_map, space_map, stem_map, iou] = analytical_gradient_combined(cuboid, P, T*inv(A), processed_depth_map, lin_ind, sampled_visible_pts, init_cuboid);
    save_results(re_depth_map, space_map, stem_map, iou, 1, count_num)
    [best_params, re_depth_map, space_map, stem_map, iou] = analytical_gradient_forward_v3(cuboid, P, T*inv(A), processed_depth_map, lin_ind, sampled_visible_pts, pts_3d_record, init_cuboid);
    save_results(re_depth_map, space_map, stem_map, iou, 2, count_num)
    % fin_param = analytical_gradient_pos_v3(cuboid, P, T*inv(A), processed_depth_map, lin_ind, sampled_visible_pts, pts_3d_record, init_cuboid, count_num, path1);
    % fin_param = analytical_gradient_v3(cuboid, P, T*inv(A), processed_depth_map, lin_ind, sampled_visible_pts, init_cuboid);
    % fin_param = analytical_gradient_inverse_1(cuboid, P, T*inv(A), processed_depth_map, lin_ind, sampled_visible_pts);
end
function print_matrix(fileID, m)
    for i = 1 : size(m,1)
        for j = 1 : size(m,2)
            fprintf(fileID, '%d\t', m(i,j));
        end
        fprintf(fileID, '\n');
    end
end
function record_metric(max_iou, count_num, flag)
    global path1 path2
    if flag == 1
        path = path1;
    end
    if flag ==2
        path = path2;
    end
    if count_num == 1
        f1 = fopen([path '/' 'iou.txt'],'w');
        print_matrix(f1, max_iou);
    else
        f1 = fopen([path '/' 'iou.txt'],'a');
        print_matrix(f1, max_iou);
    end
    fclose(f1);
end
function path = make_dir()
    global path_mul computer_flag
    computer_flag = 1;
    if computer_flag == 1
        father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/';
    else
        father_folder = '/Volumes/Untitled/Synthia_re/';
    end
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path_mul = [father_folder DateString '_mul/'];
    mkdir(path_mul); path = path_mul; mkdir([path_mul 'rgb_image/']);
end
function cuboid = mutate_cuboid(cuboid)
    params = generate_cubic_params(cuboid);
    params(1) = params(2) + rand * 0.5;
    params(4) = params(4) + abs(rand) * 2;
    params(5) = params(5) + abs(rand) * 2;
    params(2) = params(2) + abs(rand) * 5;
    params(3) = params(3) + abs(rand) * 5;
    cuboid = generate_center_cuboid_by_params(params);
end
%{
function cuboid = mutate_cuboid(cuboid)
    params = generate_cubic_params(cuboid);
    params(1) = params(1) + pi/3;
    params(2) = params(2) + 10;
    params(3) = params(3) + 10;
    cuboid = generate_center_cuboid_by_params(params);
end
%}
function new_cuboid = edit_cubic_shape(cuboid)
    params = generate_cubic_params(cuboid); [length, width] = random_length_width();
    params(4) = length; params(5) = width; params(1) = params(1) + 0.6;
    new_cuboid = generate_cuboid_by_center(params(2), params(3), params(1), params(4), params(5), params(6));
end
function [length, width] = random_length_width()
    length_base = randi([5 20], 1); width_base = randi([5 20], 1);
    length_var = rand() * 3; width_var = rand() * 3;
    length = length_base + length_var; width = width_base + width_var;
end
function params = generate_cubic_params(cuboid)
    theta = cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    params = [theta, xc, yc, l, w, h];
end
function rectangulars = stack_sampled_pts(rectangulars)
    for i = 1 : length(rectangulars)
        rectangle = rectangulars{i}; intrinsic_params = rectangle.intrinsic_params; extrinsic_params = rectangle.extrinsic_params; affine_matrix = rectangle.affine_matrx;
        rectangulars{i}.visible_pts = acquire_visible_sampled_points(rectangle, intrinsic_params, extrinsic_params, affine_matrix);
    end
end
function sampled_pts = acquire_visible_sampled_points(cuboid, intrinsic_params, extrinsic_params, affine_matrix)
    sample_pt_num = 10;
    sampled_pts = sample_cubic_by_num(cuboid, sample_pt_num, sample_pt_num);
    extrinsic_params = extrinsic_params * inv(affine_matrix);
    visible_label = find_visible_pt_global({cuboid}, sampled_pts(:, 1:3), intrinsic_params, extrinsic_params);
    sampled_pts = sampled_pts(visible_label, :);
end
function processed_depth_map = get_processed_depth_map(linear_ind, depth_map)
    max_val = max(depth_map(linear_ind));
    processed_depth_map = ones(size(depth_map)) * max_val * 1.5;
    processed_depth_map(linear_ind) = depth_map(linear_ind);
end
function draw_point_map(rectangulars)
    figure(1); clf;
    for i = 1 : length(rectangulars)
        color = rectangulars{i}.color;
        pts = rectangulars{i}.pts_new; visible_sample_pts = rectangulars{i}.visible_pts;
        scatter3(pts(:,1),pts(:,2),pts(:,3),3,color,'fill'); hold on;
        draw_cubic_shape_frame(rectangulars{i}.cur_cuboid); hold on;
        scatter3(visible_sample_pts(:,1),visible_sample_pts(:,2),visible_sample_pts(:,3),3,'r','fill');
    end
end
function img = cubic_lines_of_2d(img, cubic, intrinsic_params, extrinsic_params)
    % color = uint8(randi([1 255], [1 3]));
    % color = rand([1 3]);
    shapeInserter = vision.ShapeInserter('Shape', 'Lines', 'BorderColor', 'White');
    pts3d = zeros(8,4);
    for i = 1 : 4
        pts3d(i, :) = [cubic{i}.pts(1, :) 1];
    end
    for i = 5 : 8
        pts3d(i, :) = [cubic{5}.pts(i - 4, :) 1];
    end
    pts2d = (intrinsic_params * extrinsic_params * [pts3d(:, 1:3) ones(size(pts3d,1),1)]')';
    depth = pts2d(:,3);
    pts2d(:, 1) = pts2d(:,1) ./ depth; pts2d(:,2) = pts2d(:,2) ./ depth; pts2d = round(pts2d(:,1:2));
    lines = zeros(12, 4);
    lines(4, :) = [pts2d(4, :) pts2d(1, :)];
    lines(12, :) = [pts2d(5, :) pts2d(8, :)];
    for i = 1 : 3
        lines(i, :) = [pts2d(i, :) pts2d(i+1, :)];
    end
    for i = 1 : 4
        lines(4 + i, :) = [pts2d(i, :), pts2d(i + 4, :)];
    end
    for i = 1 : 3
        lines(8 + i, :) = [pts2d(i + 4, :) pts2d(i + 5, :)];
    end
    for i = 1 : 12
        img = step(shapeInserter, img, int32([lines(i, 1) lines(i, 2) lines(i, 3) lines(i, 4)]));
        % figure(1)
        % imshow(img)
        % pause()
    end
end
function img = render_image(img, rectangulars)
    for i = 1 : length(rectangulars)
        intrinsic_params = rectangulars{i}.intrinsic_params;
        extrinsic_params = rectangulars{i}.extrinsic_params * inv(rectangulars{i}.affine_matrx);
        color = rectangulars{i}.color;
        img = cubic_lines_of_2d(img, rectangulars{i}.cur_cuboid, intrinsic_params, extrinsic_params, color, rectangulars{i}.instanceId);
    end
end
function [depth_map, linear_ind, pts_3d_record] = generate_cubic_depth_map_by(cubic, intrinsic, extrinsic, depth_map, visible_pts)
    map_siz = size(depth_map); bias = 10;
    [projected2d, depth] = project_point_2d(extrinsic, intrinsic, visible_pts(:, 1:3)); 
    range2d_x = 200; range2d_y = 400;
    mean_position = round(mean(projected2d)); scatter2dx = mean_position(1) - range2d_x : mean_position(1) + range2d_x; scatter2dy = mean_position(1) - range2d_y : mean_position(1) + range2d_y;
    [scatter2dx, scatter2dy] = meshgrid(scatter2dx, scatter2dy); scatter2dx = scatter2dx(:); scatter2dy = scatter2dy(:);
    [visible_pts_2d, visible_pts_2d_depth, pts_3d_record] = find_visible_2d_pts({cubic}, [scatter2dx scatter2dy], intrinsic, extrinsic);
    linear_ind = sub2ind(map_siz, visible_pts_2d(:,2), visible_pts_2d(:,1)); depth_map = ones(map_siz) * (max(visible_pts_2d_depth) + bias); depth_map(linear_ind) = visible_pts_2d_depth;
    % figure(1); clf; draw_cubic_shape_frame(cubic); hold on; scatter3(pts_3d_record(:,1),pts_3d_record(:,2),pts_3d_record(:,3),3,'r','fill')
    %{
    depth_map = ones(size(depth_map)) * (max(max(depth)) + bias);
    dist = (extrinsic * [sampled_points(:,1:3) ones(size(sampled_points,1),1)]')'; dist = dist(:,1:3); dist = sum(dist.*dist, 2); [val, ind] = sort(dist); val_count = 0;
    for i = 1 : size(sampled_points, 1)
        try
            linear_ind = sub2ind(map_siz, projected2d(ind(i), 2), projected2d(ind(i), 1));
            depth_map(linear_ind) = depth(ind(i));
            val_count = val_count + 1;
        catch
        end
    end
    %}
    
    % figure(1); clf; imshow(depth_map / max(max(depth_map)));
    depth_map = imgaussfilt(depth_map,0.5);
    % figure(1); clf; imshow(depth_map / max(max(depth_map)));
end
function rectangulars = get_init_guess(objs)
    % figure(1); clf;
    rectangulars = cell(length(objs), 1);
    for i = 1 : length(objs)
        pts = objs{i}.pts_new;
        [params, cuboid] = estimate_rectangular(pts);
        rectangulars{i}.guess = params;
        rectangulars{i}.cur_cuboid = cuboid;
        rectangulars{i}.extrinsic_params = objs{i}.extrinsic_params;
        rectangulars{i}.intrinsic_params = objs{i}.intrinsic_params;
        rectangulars{i}.affine_matrx = objs{i}.affine_matrx;
        rectangulars{i}.color = objs{i}.color;
        rectangulars{i}.instanceId = objs{i}.instanceId;
        rectangulars{i}.pts_new = objs{i}.pts_new;
        rectangulars{i}.pts_old = objs{i}.pts_old;
        rectangulars{i}.linear_ind = objs{i}.linear_ind;
        % scatter3(pts(:, 1), pts(:, 2), pts(:, 3), 3, 'r', 'fill');
        % hold on; draw_cubic_shape_frame(rectangulars{i}.cur_cuboid); hold on
    end
end
function [stored_mark, extrinsic_params, intrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame)
    intrinsic_params = get_intrinsic_matrix();
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
    stored_mark = load(['/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map/', f, '.mat']); stored_mark = stored_mark.prev_mark;
end
function diff = calculate_differences(new_pts, extrinsic_params, intrinsic_params, depth_map)
    height = size(depth_map, 1); width = size(depth_map, 2);
    pts_2d = round(extrinsic_params * intrinsic_params * new_pts')';
end

function params = find_local_optimal_on_fixed_points(obj, intrinsic_params, extrinsic_params, visible_pt_3d)
    activation_label = [1 1 1 1 1 0];
    gamma = 0.5; terminate_ratio = 0.05; delta_threshold = 0.001; max_it = 100; diff_record = zeros(max_it, 1); it_count = 0;
    while it_count < max_it
        it_count = it_count + 1;
        cur_activation_label = cancel_co_activation_label(activation_label); activated_params_num = sum(double(cur_activation_label));
        hessian = zeros(activated_params_num, activated_params_num); first_order = zeros(activated_params_num, 1);
        [hessian, first_order, cur_tot_diff_record] = analytical_gradient(obj.cur_cuboid, intrinsic_params, extrinsic_params, visible_pt_3d, obj.depth_map, hessian, first_order, cur_activation_label);
        [delta, ~] = calculate_delta(hessian, first_order);
        [params_cuboid_order, ~] = update_params(obj.guess, delta, gamma, cur_activation_label, terminate_ratio);
        obj.guess(1:6) = params_cuboid_order;
        cx = params_cuboid_order(1); cy = params_cuboid_order(2); theta = params_cuboid_order(3); l = params_cuboid_order(4); w = params_cuboid_order(5); h = params_cuboid_order(6);
        obj.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
        diff_record(it_count) = cur_tot_diff_record;
        if max(abs(delta)) < delta_threshold
            break;
        end
    end
    params = obj.guess;
end
function extrinsic_params = get_new_extrinsic_params(extrinsic_params)
    load('affine_matrix.mat');
    extrinsic_params = extrinsic_params / affine_matrx;
end
function activation_label = cancel_co_activation_label(activation_label)
    activation_label = (activation_label == 1);
    if (activation_label(2) || activation_label(3)) && (activation_label(5) || activation_label(4))
        if(randi([1 2], 1) == 1)
            activation_label(2) = 0; activation_label(3) = 0;
        else
            activation_label(5) = 0; activation_label(4) = 0;
        end
    end
end
function [delta, terminate_flag] = calculate_delta(hessian, first_order)
    lastwarn(''); % Empty existing warning
    delta = hessian \ first_order;
    [msgstr, msgid] = lastwarn;
    terminate_flag = false;
    if strcmp(msgstr,'矩阵为奇异工作精度。') && strcmp(msgid, 'MATLAB:singularMatrix')
        delta = 0;
        disp('Frame Discarded due to singular Matrix, terminated')
        terminate_flag = true;
    end
end
function [params_cuboid_order, terminate_flag] = update_params(old_params, delta, gamma, activation_label, termination_ratio)
    terminate_flag = false;
    activation_label = (activation_label == 1);
    new_params = old_params;
    params_derivation_order = [new_params(3), new_params(1), new_params(2), new_params(4), new_params(5), new_params(6)];
    if max(abs(delta ./ params_derivation_order(activation_label))) < termination_ratio
        terminate_flag = true;
    end
    params_derivation_order(activation_label) = params_derivation_order(activation_label) + gamma * delta';
    params_cuboid_order = [params_derivation_order(2), params_derivation_order(3), params_derivation_order(1), params_derivation_order(4), params_derivation_order(5), params_derivation_order(6)];
    if params_cuboid_order(4) < 0 || params_cuboid_order(5) < 0 || params_cuboid_order(6) < 0
        params_cuboid_order = old_params;
        terminate_flag = true;
        disp('Impossible cubic shape, terminated')
    end
end
function cuboid = generate_center_cuboid_by_params(params)
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
end
function alternate_cuboids = transfer_cuboid(cuboid)
    alternate_cuboids = cell(4,1);
    params = generate_cubic_params(cuboid);
    for i = 1 : 4
        if i == 1
            cur_params = params;
            alternate_cuboids{i} = generate_center_cuboid_by_params(cur_params);
        end
        if i == 2
            cur_params = params; cur_params(1) = cur_params(1) + pi / 2;
            cur_params(4) = params(5); cur_params(5) = params(4);
            alternate_cuboids{i} = generate_center_cuboid_by_params(cur_params);
        end
        if i == 3
            cur_params = params; cur_params(1) = cur_params(1) + pi;
            alternate_cuboids{i} = generate_center_cuboid_by_params(cur_params);
        end
        if i == 4
            cur_params = params; cur_params(1) = cur_params(1) + pi / 2 * 3;
            cur_params(4) = params(5); cur_params(5) = params(4);
            alternate_cuboids{i} = generate_center_cuboid_by_params(cur_params);
        end
    end
end
function [cuboid, is_valid] = tune_cuboid(cuboid, extrinsic, intrinsic)
    alternate_cuboids = transfer_cuboid(cuboid);
    judge_re = false(length(alternate_cuboids),1); is_valid = false;
    for i = 1 : length(judge_re)
        judge_re(i) = jude_is_first_and_second_plane_visible(alternate_cuboids{i}, intrinsic, extrinsic);
    end
    ind = find(judge_re);
    if length(ind) > 0
        cuboid = alternate_cuboids{ind};
        is_valid = true;
    end
end
function is_visible = jude_is_first_and_second_plane_visible(cuboid, intrinsic_params, extrinsic_params)
    params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    c1 = [
        xc + 1 / 2 * w * sin(theta);
        yc - 1 / 2 * w * cos(theta);
        h/2;
        1;
        ];
    c2 = [
        xc + 1/2 * l * cos(theta);
        yc + 1/2 * l * sin(theta);
        1/2 * h;
        1
        ];
    visible_pt_label = find_visible_pt_global({cuboid}, [c1';c2'], intrinsic_params, extrinsic_params);
    is_visible = visible_pt_label(1) & visible_pt_label(2); 
end