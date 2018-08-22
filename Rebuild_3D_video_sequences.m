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

% Do not stick with the specific details of the problem
% Do Single scene reconstruction first
% Do Multiple scene reconstruction later on
% Two problem, sampling problem and a termination condition selection
% problem
function env_set()
    base_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    GT_RGB_path = 'RGB/Stereo_Left/Omni_F/';
    GT_Color_Label_path = 'GT/COLOR/Stereo_Left/Omni_F/';
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    
    focal = 532.7403520000000; cx = 640; cy = 380; % baseline = 0.8;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
    
    n = 958;
    
    for frame = 1 : 1
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
        
        
        % get_all_3d_pt(depth, extrinsic_params, intrinsic_params, label);
        objs = seg_image(depth, label, instance, extrinsic_params, intrinsic_params);
        % draw_segmented_objs(objs, img)
        objs = get_init_guess(objs);
        for i = 1 : length(objs)
            objs = estimate_single_cubic_shape(objs, extrinsic_params, intrinsic_params, i);
        end
        
        draw_scene(objs, 1, color_gt)
    end
    % Check:
    % mean_error = check_projection(objs, extrinsic_params,
    % intrinsic_params);
    % img = imread(strcat(base_path, GT_RGB_path, num2str((frame-1), '%06d'), '.png'));
end

function extrinsic_params = get_new_extrinsic_params(extrinsic_params)
    load('affine_matrix.mat');
    extrinsic_params = extrinsic_params / affine_matrx;
end
function objs = estimate_single_cubic_shape(objs, extrinsic_params, intrinsic_params, index)
    % Iinit:
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/Termination_condition_problem/';
    extrinsic_params = get_new_extrinsic_params(extrinsic_params);
    R = extrinsic_params(1:3, 1:3);
    T = extrinsic_params(1:3, 4);
    image_size = size(objs{1}.depth_map);
    camera_origin = (-R' * T)';
    num1 = 10; num2 = 10;
    gamma = 0.5; delta_threshold = 0.01;
    activation_label = [1 1 1 1 1 0];
    it_count = 0;
    is_terminated = false; terminate_ratio = 0.05; distortion_terminate_ratio = 1.1;
    max_it_num = 300; 
    tot_dist_record = zeros(max_it_num, 1); tot_params_record = zeros(max_it_num, 6);
    
    tot_dist_record(1) = calculate_ave_distance(objs{index}.cur_cuboid, objs{index}.new_pts); tot_params_record(1, :) = objs{index}.guess;
    while ~is_terminated
        it_count = it_count + 1;
        cur_activation_label = cancel_co_activation_label(activation_label);
        
        cubics = distill_all_eisting_cubic_shapes(objs);
        cur_pts = sample_cubic_by_num(objs{index}.cur_cuboid, num1, num2);
        [pts_estimated_2d, pts_estimated_vlaid, ~, depth] = projectPoints(cur_pts, intrinsic_params(1:3, 1:3), extrinsic_params, [0,0,0,0,0], [image_size(1) image_size(2)], false);
        cur_pts = cur_pts(pts_estimated_vlaid, :); pts_estimated_2d = pts_estimated_2d(pts_estimated_vlaid, :); depth = depth(pts_estimated_vlaid);
        [visible_pt_3d, ~, ~] = find_visible_pt_global(cubics, pts_estimated_2d, cur_pts, depth, intrinsic_params, extrinsic_params, camera_origin);
        
        activated_params_num = sum(double(cur_activation_label));
        hessian = zeros(activated_params_num, activated_params_num); first_order = zeros(activated_params_num, 1);
        [hessian, first_order] = analytical_gradient(objs{index}.cur_cuboid, intrinsic_params, extrinsic_params, visible_pt_3d, objs{index}.depth_map, hessian, first_order, cur_activation_label);
        
        
        figure(1)
        clf
        scatter3(visible_pt_3d(:,1), visible_pt_3d(:,2), visible_pt_3d(:,3), 3, 'r', 'fill');
        hold on
        draw_cuboid(cubics{index});
        hold on
        scatter3(objs{index}.new_pts(:,1), objs{index}.new_pts(:,2), objs{index}.new_pts(:,3), 3, 'g', 'fill')
        
        [delta, terminate_flag_singular] = calculate_delta(hessian, first_order); [params_cuboid_order, terminate_flag] = update_params(objs{index}.guess, delta, gamma, cur_activation_label, terminate_ratio);
        objs{index}.guess(1:6) = params_cuboid_order;
        cx = params_cuboid_order(1); cy = params_cuboid_order(2); theta = params_cuboid_order(3); l = params_cuboid_order(4); w = params_cuboid_order(5); h = params_cuboid_order(6);
        objs{index}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
        ave_dist = calculate_ave_distance(objs{index}.cur_cuboid, objs{index}.new_pts); tot_dist_record(it_count + 1) = ave_dist; tot_params_record(it_count + 1, :) = objs{index}.guess;
        
        if max(abs(delta)) < delta_threshold || it_count >= max_it_num || terminate_flag || terminate_flag_singular || (tot_dist_record(it_count + 1) / min(tot_dist_record(1:(it_count + 1)))) > distortion_terminate_ratio
            is_terminated = true;
        end
    end
    objs = find_best_fit_cubic(objs, tot_dist_record, tot_params_record, index);
    figure(1)
    clf
    scatter3(visible_pt_3d(:,1), visible_pt_3d(:,2), visible_pt_3d(:,3), 3, 'r', 'fill');
    hold on
    draw_cuboid(cubics{index});
    hold on
    scatter3(objs{index}.new_pts(:,1), objs{index}.new_pts(:,2), objs{index}.new_pts(:,3), 3, 'g', 'fill')
end
function objs = find_best_fit_cubic(objs, tot_dist_record, tot_params_record, index)
    selector = (tot_dist_record ~= 0); tot_dist_record = tot_dist_record(selector); tot_params_record = tot_params_record(selector, :);
    min_index = find(tot_dist_record == min(tot_dist_record));
    if length(min_index) > 1
        warning('Multiple minimum values')
        min_index = min_index(1);
    end
    params_cuboid_order = objs{index}.guess;
    cx = params_cuboid_order(1); cy = params_cuboid_order(2); theta = params_cuboid_order(3); l = params_cuboid_order(4); w = params_cuboid_order(5); h = params_cuboid_order(6);
    objs{index}.guess = tot_params_record(min_index, :);
    objs{index}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
end
function ave_dist = calculate_ave_distance(cuboid, pts)
    theta_ = -cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; bottom_center = mean(cuboid{5}.pts);
    
    transition = [
        1,  0,  0,  -bottom_center(1);
        0,  1,  0,  -bottom_center(2);
        0,  0,  1,  -bottom_center(3)/2;
        0,  0,  0,  1;
        ];
    r_on_z = [
        cos(theta_) -sin(theta_)    0   0;
        sin(theta_) cos(theta_)     0   0;
        0           0               1   0;
        0           0               0   1;
        ];
    scaling = [
        1/l,    0,      0,      0;
        0,    1/w,      0,      0;
        0,      0,      1/h,    0;
        0,      0,      0,      1;
        ];
    affine_matrix = scaling * r_on_z * transition;
    pts = [pts(:, 1:3) ones(size(pts,1), 1)]; pts = (affine_matrix * pts')';
    intern_dist = abs(pts(:,1:3)) - 0.5; intern_dist(intern_dist < 0) = 0;
    dist = sum(intern_dist.^2, 2); dist(dist == 0) = min(0.5 - abs(pts(dist == 0, 1 : 3)), [], 2);
    ave_dist = sum(dist) / size(pts, 1);
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
        objs = k_mean_check(objs, i);
        % [params, cuboid] = estimate_rectangular(objs{i}.new_pts);
        % objs{i}.guess = params;
        % objs{i}.cur_cuboid = cuboid;
    end
end
function objs = k_mean_check(objs, index)
    obj = objs{index};
    is_terminated = false;
    [next_params, ~] = estimate_rectangular(obj.new_pts);
    idx = ones(size(obj.linear_ind)); cur_volume = next_params(4) * next_params(5) * next_params(6); cur_center_num = 1;
    split_threshold = 3; min_split_num = 4;
    while true
        if is_terminated
            break
        end
        center_num_old = cur_center_num; next_params_old = next_params; idx_old = idx; 
        cur_center_num = cur_center_num + 1; next_params = zeros(cur_center_num, 6); idx = kmeans(obj.new_pts(:, 1:2),cur_center_num);
        for i = 1 : cur_center_num
            linear_ind = find(idx == i);
            if length(linear_ind) >= min_split_num
                next_params(i, :) = estimate_rectangular(obj.new_pts(linear_ind, :));
            else
                next_params(i, :) = [0, 0, 0, 0, 0, 0];
            end
        end
        
        if cur_volume / calculate_sum_volume(next_params) < split_threshold
            is_terminated = true;
        end
        cur_volume = calculate_sum_volume(next_params);
    end
    objs = split_cells(objs, index, center_num_old, next_params_old, idx_old, min_split_num);
end
function objs = split_cells(objs, index, center_num_old, next_params_old, idx_old, min_split_num)
    old_obj = objs{index};
    is_first = false;
    for i = 1 : center_num_old
        if length(idx_old == i) > min_split_num
            if ~is_first
                objs{index} = give_value_to_splitted_obj(old_obj, next_params_old, idx_old, i);
                is_first = true;
            else
                objs{end + 1} = give_value_to_splitted_obj(old_obj, next_params_old, idx_old, i);
            end
        end
    end
end
function splitted_obj = give_value_to_splitted_obj(old_obj, next_params_old, idx_old, ind)
    splitted_obj = old_obj;
    guess = next_params_old(ind, :);
    splitted_obj.guess = guess;
    splitted_obj.cur_cuboid = generate_cuboid_by_center(guess(1),guess(2),guess(3),guess(4),guess(5),guess(6));
    splitted_obj.linear_ind = old_obj.linear_ind(idx_old == ind);
    splitted_obj.old_pts = old_obj.old_pts(idx_old == ind, :);
    splitted_obj.new_pts = old_obj.new_pts(idx_old == ind, :);
    splitted_obj.depth_map = zeros(size(old_obj.depth_map));
    splitted_obj.depth_map(splitted_obj.linear_ind) = old_obj.depth_map(splitted_obj.linear_ind);
end
function volume = calculate_sum_volume(params)
    volume = 0;
    for i = 1 : size(params, 1)
        volume = volume + params(i, 4) * params(i, 5) * params(i, 6);
    end
end
function objs = seg_image(depth_map, label, instance, extrinsic_params, intrinsic_params)
    % Only for car currently;
    tot_type_num = 15; % in total 15 labelled categories
    max_depth = max(max(depth_map));
    min_obj_pixel_num = [inf, 800, inf, inf, inf, inf, 10, 10, inf, 10, 10, inf, inf, inf, inf];
    
    tot_obj_num = 0;
    objs = cell(tot_obj_num);
    
    existing_instance = unique(instance);
    labelled_pixel = false(size(instance));
    
    for i = 1 : length(existing_instance)
        cur_instance = existing_instance(i);
        if cur_instance == 0
            continue;
        end
        [ix, iy] = find(instance == cur_instance);
        linear_ind = sub2ind(size(instance), ix, iy);
        labelled_pixel(linear_ind) = true;
        
        type = label(linear_ind(1));
        instance_id = instance(linear_ind(1));
        tot_obj_num = tot_obj_num + 1;
        objs{tot_obj_num, 1} = init_single_obj(depth_map, linear_ind, extrinsic_params, intrinsic_params, type, instance_id, max_depth);
        
        instance(linear_ind) = 0;
        label(linear_ind) = 0;
    end
    
    % Exclude void(0), reserved1(13), reserved2(14), sky(1), tree(6)
    % road(3), sidewalk(4), fence(5), lanemarking(12)
    for i = 1 : tot_type_num
        if i ~= 2 && i ~= 7 && i ~= 8 && i ~= 10 && i ~= 11 && i ~= 6
            continue
        end
        [ix, iy] = find(label == i);
        linear_ind = sub2ind(size(instance), ix, iy);
        if ~isempty(linear_ind)
            binary_map = false(size(instance));
            binary_map(linear_ind) = true;
            CC = bwconncomp(binary_map);
            for j = 1 : CC.NumObjects
                if length(CC.PixelIdxList{j}) > min_obj_pixel_num(i)
                    tot_obj_num = tot_obj_num + 1;
                    objs{tot_obj_num, 1} = init_single_obj(depth_map, CC.PixelIdxList{j}, extrinsic_params, intrinsic_params, i, 0, max_depth);
                end
                label(CC.PixelIdxList{j}) = 0;
                instance(CC.PixelIdxList{j}) = 0;
            end
        end
    end
    
    if max(max(label)) ~= 0 || max(max(instance)) ~= 0
        warning('Some objects not distilled')
    end
    
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
end

function draw_scene(objs, index, color_gt)
    cmap = colormap;
    new_color_gt = uint8(zeros(size(color_gt)));
    figure(index)
    clf
    for i = 1 : length(objs)
        % if objs{i}.type ~= 7
        %     continue
        % end
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