clear; clc;
max_frame = 294; frame = 1; s_frame = 50; e_frame = 60; instance_id = 1;
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
function org_entry = read_in_org_entry(frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_unioned/';
    ind = num2str(frame, '%06d');
    loaded = load([path ind '.mat']); org_entry = loaded.prev_mark;
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
    the_obj.adjust_matrix = org_obj.adjust_matrix;
    the_obj.affine_matrx = org_obj.affine_matrx;
    the_obj.extrinsic_params = the_obj.extrinsic_params * inv(frame_tune_matrix);
    the_obj.pts_old = (frame_tune_matrix * the_obj.pts_old')';
    the_obj.pts_new = (the_obj.affine_matrx * the_obj.pts_old')';
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
function save_results(re_depth_map, space_map, stem_map, iou, flag, count_num)
    global path1 path2
    record_metric(iou, count_num, flag);
    if flag == 1
        imwrite(re_depth_map, [path1 '/' num2str(count_num) '_' '2d_img' '.png']);
        imwrite(space_map, [path1 '/' num2str(count_num) '_' '3d_img' '.png']);
        imwrite(stem_map, [path1 '/' num2str(count_num) '_' 'stem_img' '.png']);
    end
    if flag == 2
        imwrite(re_depth_map, [path2 '/' num2str(count_num) '_' '2d_img' '.png']);
        imwrite(space_map, [path2 '/' num2str(count_num) '_' '3d_img' '.png']);
        imwrite(stem_map, [path2 '/' num2str(count_num) '_' 'stem_img' '.png']);
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
    father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/';
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path = [father_folder DateString '/multiple_frame_based/'];
    mkdir(path);
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