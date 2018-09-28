clear; clc;
global exp_num
exp_num = 1;
max_frame = 294; % frame = randi([1 294], 1);
frame = 1;
[mark, extrinsic_params, intrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame);
rectangulars = get_init_guess(mark); optimize_rectangles(rectangulars, depth);
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
function optimize_rectangles(rectangulars, depth_map)
    it_num = 100;
    for i = 1 : it_num
        j = 1;
        optimize_cubic_shape(rectangulars{j}, depth_map);
    end
end
function optimize_cubic_shape(rectangular, depth_map)
    %{
    cuboid = rectangular.cur_cuboid; P = rectangular.intrinsic_params; A = rectangular.affine_matrx; T = rectangular.extrinsic_params;
    cuboid = edit_cubic_shape(cuboid); rectangular.cur_cuboid = cuboid;
    sampled_visible_pts = acquire_visible_sampled_points(rectangular, P, T, A); a = sampled_visible_pts;
    processed_depth_map = generate_cubic_depth_map_by(cuboid, P, T*inv(A), depth_map, sampled_visible_pts);
    sampled_visible_pts = [sampled_visible_pts(:, 5:6) sampled_visible_pts(:, 4)]; projected_sampled_points = project_point_2d(T, P, a(:,1:3), A);
    %}
    load('trail.mat')
    fin_param = analytical_gradient_v3(cuboid, P, T*inv(A), sampled_visible_pts, processed_depth_map, rectangular.pts_new, projected_sampled_points);
end
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
function sampled_pts = acquire_visible_sampled_points(rectangle, intrinsic_params, extrinsic_params, affine_matrix)
    cuboid = rectangle.cur_cuboid; sample_pt_num = 10;
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
function img = cubic_lines_of_2d(img, cubic, intrinsic_params, extrinsic_params, color, instance_id)
    % color = uint8(randi([1 255], [1 3])); 
    % color = rand([1 3]);
    % shapeInserter = vision.ShapeInserter('Shape', 'Lines', 'BorderColor', color);
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
        % img = step(shapeInserter, img, int32([lines(i, 1) lines(i, 2) lines(i, 3) lines(i, 4)]));
        img = insertShape(img,'Line', int32([lines(i, 1) lines(i, 2) lines(i, 3) lines(i, 4)]), 'LineWidth', 2, 'Color', ceil(color * 254));
        % pause()
    end
    text_position_ind = find(sum(pts2d, 2) == min(sum(pts2d, 2))); str = ['ins: ' num2str(instance_id)];
    img = insertText(img, pts2d(text_position_ind, :) - [20 10], str, 'FontSize', 10, 'BoxColor', ceil(color * 254),'BoxOpacity',0.4,'TextColor','white');
end
function img = render_image(img, rectangulars)
    for i = 1 : length(rectangulars)
        intrinsic_params = rectangulars{i}.intrinsic_params;
        extrinsic_params = rectangulars{i}.extrinsic_params * inv(rectangulars{i}.affine_matrx);
        color = rectangulars{i}.color;
        img = cubic_lines_of_2d(img, rectangulars{i}.cur_cuboid, intrinsic_params, extrinsic_params, color, rectangulars{i}.instanceId);
    end
end
function depth_map = generate_cubic_depth_map_by(cubic, intrinsic, extrinsic, depth_map, visible_pts)
    map_siz = size(depth_map); depth_map = ones(map_siz) * 300;
    [projected2d, depth] = project_point_2d(extrinsic, intrinsic, visible_pts(:, 1:3)); 
    range2d_x = 200; range2d_y = 400;
    mean_position = round(mean(projected2d)); scatter2dx = mean_position(1) - range2d_x : mean_position(1) + range2d_x; scatter2dy = mean_position(1) - range2d_y : mean_position(1) + range2d_y;
    [scatter2dx, scatter2dy] = meshgrid(scatter2dx, scatter2dy); scatter2dx = scatter2dx(:); scatter2dy = scatter2dy(:);
    [visible_pts_2d, visible_pts_2d_depth] = find_visible_2d_pts({cubic}, [scatter2dx scatter2dy], intrinsic, extrinsic);
    linear_ind = sub2ind(map_siz, visible_pts_2d(:,2), visible_pts_2d(:,1)); depth_map(linear_ind) = visible_pts_2d_depth;
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
    
    figure(1); clf; imshow(depth_map / max(max(depth_map)));
    depth_map = imgaussfilt(depth_map,2);
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
% Check:
% max(abs(depth_map(linear_ind)) - projected_depth)
% projected_depth = (intrinsic_params * extrinsic_params * new_pts')';
% projected_depth = projected_depth(:, 3);
% figure(1)
% clf
% draw_cuboid(objs{1}.cur_cuboid)
% hold on
% scatter3(obj.new_pts(:,1),obj.new_pts(:,2),obj.new_pts(:,3),3,'r','fill')