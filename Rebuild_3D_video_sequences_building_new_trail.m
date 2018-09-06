env_set()
% Label info below
% Class         ID
% Void          0
% Sky           1
% Building      2
% Road          3
% Sidewalk      4
% Fence         5
% Vegetation    6
% Pole          7
% Car           8
% Traffic Sign  9
% Pedestrian    10
% Bicycle       11
% Lanemarking	12
% Reserved		13
% Reserved      14
% Traffic Light	15
function env_set()
    base_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    GT_RGB_path = 'RGB/Stereo_Left/Omni_F/';
    GT_Color_Label_path = 'GT/COLOR/Stereo_Left/Omni_F/';
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    
    focal = 532.7403520000000; cx = 640; cy = 380; % baseline = 0.8;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
    
    n = 294; tot_obj_dist = 0; tot_obj_diff = 0; tot_dist = 0; tot_diff = 0;
    
    for frame = 1 : 1
        affine_matrx = Estimate_ground_plane(frame); save('affine_matrix.mat', 'affine_matrx');
        f = num2str(frame, '%06d');
        
        color_gt = imread(strcat(base_path, GT_Color_Label_path, num2str((frame-1), '%06d'), '.png'));
        
        % Get Camera parameter
        txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt');
        vec = load(txtPath);
        extrinsic_params = reshape(vec, 4, 4);
        
        % Get Depth groundtruth
        ImagePath = strcat(base_path, GT_Depth_path, f, '.png');
        depth = getDepth(ImagePath);
        
        % Get segmentation mark groudtruth (Instance id looks broken)
        ImagePath = strcat(base_path, GT_seg_path, f, '.png');
        [label, instance] = getIDs(ImagePath);
        
        ImagePath = strcat(base_path, GT_RGB_path, f, '.png');
        img = imread(ImagePath);
        
        % get_all_3d_pt(depth, extrinsic_params, intrinsic_params, label);
        objs = seg_image(depth, label, instance, extrinsic_params, intrinsic_params);
        % draw_segmented_objs(objs, img)
        objs = get_init_guess(objs);
        for i = 1 : length(objs)
            objs = estimate_single_cubic_shape(objs, extrinsic_params, intrinsic_params, i, depth);
        end
        for i = 1 : length(objs)
            img = cubic_lines_of_2d(img, objs{i}.cur_cuboid, objs{i}.intrinsic_params, objs{i}.extrinsic_params);
        end
        save_img(img, frame)
        % draw_scene(objs, 1, color_gt);
        %{
        for i = 1 : length(objs)
            objs{i}.metric = calculate_metric(objs{i}, depth);
            if ~isnan(objs{i}.metric(2)) tot_dist = tot_dist + objs{i}.metric(2); tot_obj_dist = tot_obj_dist + 1; end
            if ~isnan(objs{i}.metric(1)) tot_diff = tot_diff + objs{i}.metric(1); tot_obj_diff = tot_obj_diff + 1; end
            img = cubic_lines_of_2d(img, objs{i}.cur_cuboid, objs{i}.intrinsic_params, objs{i}.extrinsic_params);
        end
        %}
        path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/Car_reconstruction_results/metric.txt';
        save_to_text(objs, frame, path);
        % save_img(img, frame)
        % disp(['Frame ' num2str(frame) ' Finished\n'])
        % figure(1)
        % clf
        % imshow(img)
        
    end
    % ave_dist = tot_dist / tot_obj_dist; ave_diff = tot_diff / tot_obj_diff;
    % save_mean_to_text(ave_dist, ave_diff, path)
    % Check:
    % mean_error = check_projection(objs, extrinsic_params,
    % intrinsic_params);
    % img = imread(strcat(base_path, GT_RGB_path, num2str((frame-1), '%06d'), '.png'));
end
function [extrinsic_params, intrinsic_params, depth, label, instance, img] = acquire_img(num_frame)
    base_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    GT_RGB_path = 'RGB/Stereo_Left/Omni_F/';
    GT_Color_Label_path = 'GT/COLOR/Stereo_Left/Omni_F/';
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    
    focal = 532.7403520000000; cx = 640; cy = 380; % baseline = 0.8;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
    frame = num_frame;
    f = num2str(frame, '%06d');
    % Get Camera parameter
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt');
    vec = load(txtPath);
    extrinsic_params = reshape(vec, 4, 4);
    
    % Get Depth groundtruth
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png');
    depth = getDepth(ImagePath);
    
    % Get segmentation mark groudtruth (Instance id looks broken)
    ImagePath = strcat(base_path, GT_seg_path, f, '.png');
    [label, instance] = getIDs(ImagePath);
    
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png');
    img = imread(ImagePath);
end
function save_img(img, frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/Car_reconstruction_results/';
    f = num2str(frame, '%06d');
    imwrite(img, [path f '.png']);
end
function save_mean_to_text(ave_dist, ave_diff, path)
    fileID = fopen(path,'a');
    fprintf(fileID,'Average Distance:\t%5d\n',ave_dist);
    fprintf(fileID,'Average Difference:\t%5d\n',ave_diff);
    fclose(fileID);
end
function save_to_text(objs, frame_num, path)
    fileID = fopen(path,'a');
    for i = 1 : length(objs)
        fprintf(fileID,'Frame_num:\t%2d\n',frame_num);
        fprintf(fileID,'3D Distance:\t%5d\n',objs{i}.metric(1));
        fprintf(fileID,'Depth Difference:\t%5d\n',objs{i}.metric(2));
    end
    fclose(fileID);
end
function extrinsic_params = get_new_extrinsic_params(extrinsic_params)
    load('affine_matrix.mat');
    extrinsic_params = extrinsic_params / affine_matrx;
end
function objs = estimate_single_cubic_shape(objs, extrinsic_params, intrinsic_params, index, depth_map)
    % Iinit:
    % path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/Termination_condition_problem/';
    extrinsic_params = get_new_extrinsic_params(extrinsic_params);
    R = extrinsic_params(1:3, 1:3);
    T = extrinsic_params(1:3, 4);
    image_size = size(objs{1}.depth_map);
    camera_origin = (-R' * T)';
    num1 = 10; num2 = 10;
    % gamma = 0.5; delta_threshold = 0.01;
    % activation_label = [1 1 1 1 1 0];
    % it_count = 0;
    is_terminated = false; % terminate_ratio = 0.05; distortion_terminate_ratio = 1.1;
    % max_it_num = 300; 
    init_guess = objs{index}.guess;
    percentage = [0.01 0.1 0.2 0.3 0.5 0.7 1];
    tot_params_record = zeros(length(percentage) + 1, 6); tot_params_record(1, :) = objs{index}.guess;
    % tot_dist_record = zeros(max_it_num, 1); 
    % tot_dist_record(1) = calculate_ave_distance(objs{index}.cur_cuboid, objs{index}.new_pts); 
    % visible_pts_record = cell(length(percentage) + 1, 1); 
    while ~is_terminated
        % Currently still implement in this way to preserve compatibility
        % with the old version
        % it_count = it_count + 1;
        % cur_activation_label = cancel_co_activation_label(activation_label);
        % cubics = distill_all_eisting_cubic_shapes(objs);
        % cubics = {objs{index}.cur_cuboid};
        cur_pts = sample_cubic_by_num(objs{index}.cur_cuboid, num1, num2);
        [pts_estimated_2d, pts_estimated_vlaid, ~, depth] = projectPoints(cur_pts, intrinsic_params(1:3, 1:3), extrinsic_params, [0,0,0,0,0], [image_size(1) image_size(2)], false);
        cur_pts = cur_pts(pts_estimated_vlaid, :); pts_estimated_2d = pts_estimated_2d(pts_estimated_vlaid, :); depth = depth(pts_estimated_vlaid);
        % [visible_pt_3d, ~, ~] = find_visible_pt_global(cubics, pts_estimated_2d, cur_pts, depth, intrinsic_params, extrinsic_params, camera_origin);
        [visible_pt_3d, ~, ~] = find_visible_pt_global({objs{index}.cur_cuboid}, pts_estimated_2d, cur_pts, depth, intrinsic_params, extrinsic_params, camera_origin);
        plane_inds = unique(visible_pt_3d(:, end)); plane_inds = plane_inds(plane_inds ~= 5);
        % objs{index}.visible_pt = visible_pt_3d;
        % visible_pts_record{1} = visible_pt_3d;
        % objs{index}.visible_pt = visible_pt_3d;
        % activated_params_num = sum(double(cur_activation_label));
        % hessian = zeros(activated_params_num, activated_params_num); first_order = zeros(activated_params_num, 1);
        % [hessian, first_order] = analytical_gradient(objs{index}.cur_cuboid, intrinsic_params, extrinsic_params, visible_pt_3d, objs{index}.depth_map, hessian, first_order, cur_activation_label);
        correspondence_last = zeros(0); last_guess = init_guess;
        for  i = 1 : length(percentage)
            correspondence = find_correspondence(objs{index}.cur_cuboid, objs{index}.new_pts, visible_pt_3d, percentage(i), num1, num2);
            selector = (correspondence ~= 0);  
            if sum(selector) == 0 || isequal(correspondence, correspondence_last)
                tot_params_record(i + 1, :) = last_guess;
                % visible_pts_record{i + 1} = visible_pts_record{i};
            else
                fin_param = analytical_gradient_v2(objs{index}.cur_cuboid, intrinsic_params, extrinsic_params, visible_pt_3d(selector,:), objs{index}.depth_map, objs{index}.new_pts(correspondence(selector), :), objs{index}.new_pts);
                tot_params_record(i + 1, :) = fin_param;
                % visible_pts_record{i + 1} = visible_pt_3d(selector,:);
            end
            last_guess = tot_params_record(i + 1, :);
            correspondence_last = correspondence;
        end
        
        metric_record = zeros(size(tot_params_record, 1), 2);
        for i = 1 : size(tot_params_record, 1)
            loc_obj = objs{index};
            loc_obj.guess = tot_params_record(i, :); fin_param = loc_obj.guess;
            loc_obj.correspondence = correspondence_last;
            cx = fin_param(1); cy = fin_param(2); theta = fin_param(3); l = fin_param(4); w = fin_param(5); h = fin_param(6);
            loc_obj.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
            loc_obj.visible_pt = sample_pts_plane(loc_obj.cur_cuboid, num1, num2, plane_inds);
            metric_record(i, :)= calculate_metric(loc_obj, depth_map);
        end
        
        is_terminated = true;
        %{
        figure(1)
        clf
        scatter3(visible_pt_3d(:,1), visible_pt_3d(:,2), visible_pt_3d(:,3), 3, 'r', 'fill');
        hold on
        draw_cuboid(cubics{index});
        hold on
        scatter3(objs{index}.new_pts(:,1), objs{index}.new_pts(:,2), objs{index}.new_pts(:,3), 3, 'g', 'fill')
        %}
        
        % [delta, terminate_flag_singular] = calculate_delta(hessian, first_order); [params_cuboid_order, terminate_flag] = update_params(objs{index}.guess, delta, gamma, cur_activation_label, terminate_ratio);
        % objs{index}.guess(1:6) = params_cuboid_order;
        % objs{index}.guess(1:6) = fin_param;
        % cx = fin_param(1); cy = fin_param(2); theta = fin_param(3); l = fin_param(4); w = fin_param(5); h = fin_param(6);
        % objs{index}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
        % ave_dist = calculate_ave_distance(objs{index}.cur_cuboid, objs{index}.new_pts); tot_dist_record(it_count + 1) = ave_dist; tot_params_record(it_count + 1, :) = objs{index}.guess;
        % is_terminated = true;
        % if max(abs(delta)) < delta_threshold || it_count >= max_it_num || terminate_flag || terminate_flag_singular || (tot_dist_record(it_count + 1) / min(tot_dist_record(1:(it_count + 1)))) > distortion_terminate_ratio
            % is_terminated = true;
        % end
    end
    objs = find_best_fit_cubic(objs, metric_record, tot_params_record, index);
    %{
    figure(1)
    clf
    scatter3(visible_pt_3d(:,1), visible_pt_3d(:,2), visible_pt_3d(:,3), 3, 'r', 'fill');
    hold on
    draw_cuboid(cubics{index});
    hold on
    scatter3(objs{index}.new_pts(:,1), objs{index}.new_pts(:,2), objs{index}.new_pts(:,3), 3, 'g', 'fill')
    %}
end
function pts = sample_pts_plane(cuboid, num1, num2, plane_num)
    pts = sample_cubic_by_num(cuboid, num1, num2);
    selector = false(size(pts, 1), 1);
    for i = 1 : length(plane_num)
        selector = (selector | pts(:, 4) == plane_num(i));
        % selector = pts(:, 4) == plane_num1 | (pts(:, 4) == plane_num2);
    end
    pts = pts(selector, :);
end
function objs = find_best_fit_cubic(objs, metric_record, tot_params_record, index)
    metric_record_cpy = metric_record; metric_record = sum(metric_record, 2);
    % selector = (metric_record ~= 0); metric_record = metric_record(selector); tot_params_record = tot_params_record(selector, :);
    min_index = find(metric_record == min(metric_record)); min_index = min_index(1);
    params_cuboid_order = tot_params_record(min_index, :);
    try
        cx = params_cuboid_order(1); cy = params_cuboid_order(2); theta = params_cuboid_order(3); l = params_cuboid_order(4); w = params_cuboid_order(5); h = params_cuboid_order(6);
        objs{index}.guess = tot_params_record(min_index, :);
        objs{index}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
        objs{index}.metric = metric_record_cpy;
        objs{index}.ind = min_index;
    catch
        disp('Error occurred')
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
function cubics = distill_all_eisting_cubic_shapes(objs)
    cubics = cell(length(objs), 1);
    for i = 1 : length(objs)
        cubics{i} = objs{i}.cur_cuboid;
    end
end
function objs = get_init_guess(objs)
    for i = 1 : length(objs)
        [params, cuboid] = estimate_rectangular(objs{i}.new_pts);
        objs{i}.guess = params;
        objs{i}.cur_cuboid = cuboid;
    end
end
function objs = seg_image(depth_map, label, instance, extrinsic_params, intrinsic_params)
    load('affine_matrix.mat')
    tot_type_num = 15; % in total 15 labelled categories
    max_depth = max(max(depth_map));
    min_obj_pixel_num = [inf, 800, inf, inf, inf, 70, 10, 10, inf, 10, 10, inf, inf, inf, inf];
    min_obj_height = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    max_obj_height = [inf, inf, inf, inf, inf, 0.10, 0.42, inf, inf, inf, inf, inf, inf, inf, inf];
    
    tot_obj_num = 0;
    objs = cell(tot_obj_num);
    
    existing_instance = unique(instance);
    labelled_pixel = false(size(instance));
    
    frame_num = 294; building_type = 2;
    
    linear_ind_record = cell(frame_num, 1); pts_3d_old_record = cell(frame_num, 1); pts_3d_new_record = cell(frame_num, 1);
    for i = 1 : frame_num
        [extrinsic_params, intrinsic_params, depth_map, label, instance, img] = acquire_img(i);
        [ix, iy] = find(label == building_type);
        linear_ind_record{i} = sub2ind(size(instance), ix, iy); 
        pts_3d_old_record{i} = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, linear_ind_record{i});
        pts_3d_new_record{i} = (affine_matrx * (pts_3d_old_record{i})')';
    end
    
    tot_num = 0;
    for i = 1 : frame_num
        tot_num = tot_num + size(pts_3d_new_record{i}, 1);
    end
    pts_3d_new_tot = zeros(tot_num, 4); num_count = 0;
    for i = 1 : frame_num
        pts_3d_new_tot(num_count + 1 : num_count + size(pts_3d_new_record{i}, 1), :) = pts_3d_new_record{i};
        num_count = num_count + size(pts_3d_new_record{i}, 1);
    end
    figure(1)
    clf
    scatter3(pts_3d_new_tot(:,1), pts_3d_new_tot(:,2), pts_3d_new_tot(:, 3), 3, 'r', 'fill')
    
     % label = eliminate_type_pixel(label, min_obj_height, max_obj_height, extrinsic_params, intrinsic_params, depth_map, building_type);
    for i = 1 : tot_type_num
        % Deal with tree
        if i ~= building_type
            continue
        end
        [ix, iy] = find(label == i);
        linear_ind = sub2ind(size(instance), ix, iy);
    end
end
function metric = calculate_metric(obj, depth_map)
    % One term is the distance from the points to the cubic shape
    % The other term is the differences between the depth 
    pts = obj.new_pts;
    cuboid = obj.cur_cuboid;
    ave_dist = calculate_ave_distance(cuboid, pts);
    ave_diff = calculate_depth_diff(depth_map, obj.visible_pt, obj.extrinsic_params, obj.intrinsic_params, obj.correspondence);
    metric = [ave_dist ave_diff];
end
function ave_diff = calculate_depth_diff(depth_map, pts_cubic, extrinsic_params, intrinsic_params, correspondence)
    % selector = (correspondence ~= 0);
    selector = true(length(correspondence), 1);
    [pts2d, depth] = get_2dloc_and_depth(pts_cubic(selector, :), extrinsic_params, intrinsic_params, size(depth_map));
    linear_ind = sub2ind(size(depth_map), pts2d(:,2), pts2d(:,1));
    gt_depth = depth_map(linear_ind);
    ave_diff = sum(abs(depth - gt_depth)) / sum(selector);
    % Code to check:
    %{
    depth_map_copy = depth_map; linear_ind = sub2ind(size(depth_map), pts2d(:,2), pts2d(:,1));
    depth_map_copy(linear_ind) = 20;
    figure(4)
    clf
    show_depth_map(depth_map_copy * 10);
    %}
end
function [pts2d, depth] = get_2dloc_and_depth(pts, extrinsic_params, intrinsic_params, img_size)
    pts2d = (intrinsic_params * extrinsic_params * [pts(:, 1:3) ones(size(pts,1),1)]')';
    depth = pts2d(:,3);
    pts2d(:,1) = pts2d(:,1) ./ depth; pts2d(:,2) = pts2d(:,2) ./ depth;
    pts2d = round(pts2d(:,1:2));
    selector = (pts2d <= 0); pts2d(selector) = 1;
    selector = (pts2d(:,1) > img_size(2)); pts2d(selector, 1) = img_size(2);
    selector = (pts2d(:,2) > img_size(1)); pts2d(selector, 2) = img_size(1);
end
function objs = further_seg(extrinsic_params, intrinsic_params, image_size, objs, ind, SE, min_pixel_num)
    pts_3d = objs{ind}.new_pts;
    xmin = min(pts_3d(:,1)); xmax = max(pts_3d(:,1)); ymin = min(pts_3d(:,2)); ymax = max(pts_3d(:,2));
    rangex = xmax - xmin; rangey = ymax - ymin;
    bimg = false(image_size);
    pixel_coordinate_x = (pts_3d(:,1) - xmin) / (rangex / (image_size(1) - 1)); pixel_coordinate_x = round(pixel_coordinate_x) + 1;
    pixel_coordinate_y = (pts_3d(:,2) - ymin) / (rangey / (image_size(2) - 1)); pixel_coordinate_y = round(pixel_coordinate_y) + 1;
    linear_ind = sub2ind(image_size, pixel_coordinate_y, pixel_coordinate_x);
    bimg(linear_ind) = true;
    J = imdilate(bimg,SE); J = imerode(J,SE);
    CC = bwconncomp(J);

    depth_map = objs{ind}.depth_map; type = objs{ind}.type; max_depth = max(max(depth_map)); tot_linear_ind = objs{ind}.linear_ind;
    objs(ind) = [];
    for i = 1 : CC.NumObjects
        img_ind = CC.PixelIdxList{i};
        cur_indices = zeros(0);
        for j = 1 : length(img_ind)
            cur_indices = [cur_indices; find(linear_ind == img_ind(j))];
        end
        if length(cur_indices) > min_pixel_num
            cur_linear_ind = tot_linear_ind(cur_indices);
            objs{end + 1, 1} = init_single_obj(depth_map, cur_linear_ind, extrinsic_params, intrinsic_params, type, 0, max_depth);
        end
    end
end
function label = eliminate_type_pixel(label, min_obj_height, max_obj_height, extrinsic_params, intrinsic_params, depth_map, type)
    min_height = min_obj_height(type); max_height = max_obj_height(type);
    [ix, iy] = find(label == type);
    linear_ind = sub2ind(size(label), ix, iy);
    old_pts = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, linear_ind);
    new_pts = get_pt_on_new_coordinate_system(old_pts);
    selector = (new_pts(:, 3) < min_height) | (new_pts(:, 3) > max_height);
    label(linear_ind(selector)) = 0;
end

function obj = init_single_obj(depth_map, linear_ind, extrinsic_params, intrinsic_params, type, instance_id, max_depth)
    obj = struct;
    obj.instance = instance_id;
    obj.type = type;
    
    obj.linear_ind = linear_ind;
    obj.depth_map = ones(size(depth_map)) * max_depth;
    obj.depth_map(obj.linear_ind) = depth_map(obj.linear_ind);
    
    obj.old_pts = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, obj.linear_ind);
    obj.new_pts = get_pt_on_new_coordinate_system(obj.old_pts);
    obj.extrinsic_params = get_new_extrinsic_params(extrinsic_params); obj.intrinsic_params = intrinsic_params;
end

function draw_scene(objs, fig_index, color_gt)
    cmap = colormap;
    new_color_gt = uint8(zeros(size(color_gt)));
    figure(fig_index)
    clf
    for i = 1 : length(objs)
        color = cmap(randi([1 64]), :);
        pts = objs{i}.new_pts;
        [I,J] = ind2sub(size(objs{i}.depth_map),objs{i}.linear_ind);
        for k = 1 : length(objs{i}.linear_ind)
            color_gt(I(k), J(k), :) = uint8([256 256 256]);
            new_color_gt(I(k), J(k), :) = uint8(round(color * 255));
        end
        scatter3(pts(:,1), pts(:,2), pts(:,3), 3, color, 'fill')
        hold on
        draw_cuboid(objs{i}.cur_cuboid)
    end
    axis equal
    figure(2)
    imshow(color_gt)
    figure(3)
    imshow(new_color_gt)
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
function show_depth_map(depth_map)
    imshow(uint16(depth_map * 1000));
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
function correspondence = find_correspondence(cuboid, gt_pts, visible_pts, percentage, num1, num2)
    % figure(1)
    % scatter3(gt_pts(:,1),gt_pts(:,2),gt_pts(:,3),5,'r');
    % hold on
    visible_plane = false(4, 1);
    for i = 1 : 4
        indices = find(visible_pts(:, 6) == i);
        if length(indices) >= 1
            visible_plane(i) = true;
        end
    end
    
    max_height = zeros(4, 1); x_y_plane = zeros(4, 2);
    for i = 1 : 4
        max_height(i) = cuboid{mod(i, 4) + 1}.length1;
        x_y_plane(i, 1) = cuboid{i}.length1; x_y_plane(i, 2) = cuboid{i}.length2; 
    end
    correspondence = zeros(size(visible_pts, 1), 1);
    for i = 1 : 4
        plane_selector = (visible_pts(:, 6) == i);
        if(visible_plane(i))
            cur_max_height = max_height(i) * percentage;
            
            T = cuboid{i}.Tflat; x_unit = x_y_plane(i, 1) / num1 / 2; y_unit = x_y_plane(i, 2) / num2 / 2;
            
            
            gt_pts_transmitted = (T * [gt_pts(:, 1:3) ones(size(gt_pts, 1), 1)]')';
            visible_pts_transmitted = (T * [visible_pts(:, 1:3) ones(size(visible_pts, 1), 1)]')';
            valid_gt_pts_transmitted = gt_pts_transmitted((abs(gt_pts_transmitted(:, 3)) < cur_max_height), :);
            valid_gt_pts = gt_pts((abs(gt_pts_transmitted(:, 3)) < cur_max_height), :);
            
            global_ind = 1 : size(gt_pts, 1); local_ind = global_ind((abs(gt_pts_transmitted(:, 3)) < cur_max_height));
            
            valid_visible_pts_transmitted = visible_pts_transmitted(visible_pts(:, 6) == i, :);
            valid_visible_pts = visible_pts(visible_pts(:, 6) == i, :);
            local_correspondence = zeros(size(valid_visible_pts_transmitted, 1), 1);
            
            
            for j = 1 : size(valid_visible_pts_transmitted, 1)
                search_range = [
                    valid_visible_pts_transmitted(j, 1) - x_unit, valid_visible_pts_transmitted(j, 1) + x_unit;
                    valid_visible_pts_transmitted(j, 2) - y_unit, valid_visible_pts_transmitted(j, 2) + y_unit;
                    ];
                slected =   valid_gt_pts_transmitted(:,1) > search_range(1,1) & ... 
                            valid_gt_pts_transmitted(:,1) < search_range(1,2) & ...
                            valid_gt_pts_transmitted(:,2) > search_range(2,1) & ...
                            valid_gt_pts_transmitted(:,2) < search_range(2,2);
                local_selected_pts = valid_gt_pts(slected, 1 : 3);
                if isempty(local_selected_pts)
                    continue
                end
                local_local_ind = local_ind(slected);
                
                dist = sum((valid_visible_pts(j, 1 : 3) - local_selected_pts).^2, 2);
                min_dist_ind = find(dist == min(dist)); min_dist_ind = min_dist_ind(1);
                local_correspondence(j) = local_local_ind(min_dist_ind);
            end
            correspondence(plane_selector) = local_correspondence;
        end
    end
end
