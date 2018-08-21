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
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    
    focal = 532.7403520000000; cx = 640; cy = 380; % baseline = 0.8;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
    
    n = 958;
    
    for frame = 1 : 1
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
        img = imread(strcat(base_path, GT_RGB_path, num2str((frame-1), '%06d'), '.png'));
        
        objs = seg_image(depth, label, instance, extrinsic_params, intrinsic_params);
        objs = get_init_guess(objs);
        objs = estimate_single_cubic_shape(objs, extrinsic_params, intrinsic_params, 1);
        
        draw_scene(objs, 1)
    end
    % Check:
    % mean_error = check_projection(objs, extrinsic_params,
    % intrinsic_params);
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
    is_terminated = false;
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
        [visible_pt_3d, visible_pt_2d, visible_depth] = find_visible_pt_global(cubics, pts_estimated_2d, cur_pts, depth, intrinsic_params, extrinsic_params, camera_origin);
        
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
        
        delta = calculate_delta(hessian, first_order); params_cuboid_order = update_params(objs{index}.guess, delta, gamma, cur_activation_label);
        objs{index}.guess(1:6) = params_cuboid_order;
        cx = params_cuboid_order(1); cy = params_cuboid_order(2); theta = params_cuboid_order(3); l = params_cuboid_order(4); w = params_cuboid_order(5); h = params_cuboid_order(6);
        objs{index}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
        ave_dist = calculate_ave_distance(objs{index}.cur_cuboid, objs{index}.new_pts); tot_dist_record(it_count + 1) = ave_dist; tot_params_record(it_count + 1, :) = objs{index}.guess;
        
        if max(abs(delta)) < delta_threshold | it_count >= max_it_num
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
    params = zeros(5, 4);
    for i = 1 : 5
        params(i, :) = cuboid{i}.params;
    end
    dist = pts * params' ./ repmat(sum(params.^2, 2)', [size(pts, 1) 1]);
    [val, loc] = min(dist');
    ave_dist = sum(val) / size(pts, 1);
    % Check:
    %{
    cmap = colormap;
    rand_color_ind = [1 13 25 37 49];
    colors = cmap(rand_color_ind, :);
    for i = 1 : 5
        selector = (loc == i);
        scatter3(pts(selector,1), pts(selector,2), pts(selector,3), 3, colors(i, :))
        hold on
    end
    %}
end
function delta = calculate_delta(hessian, first_order)
    warning(''); % Empty existing warning
    delta = hessian \ first_order;
    [msgstr, msgid] = lastwarn;
    if strcmp(msgstr,'矩阵为奇异工作精度。') & strcmp(msgid, 'MATLAB:singularMatrix')
        delta = 0;
        disp('Frame Discarded due to singular Matrix')
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
    % Only for car currently;
    [car_ix, car_iy] = find(label == 8);
    
    all_car_pixel = sub2ind(size(label), car_ix, car_iy);
    to_seg = unique(instance(all_car_pixel));
    objs = cell(length(to_seg), 1);
    
    max_depth = max(max(depth_map));
    
    for i = 1 : length(to_seg)
        objs{i} = struct;
        objs{i}.instance = to_seg(i);
        objs{i}.label = 8;
        
        [all_cur_instance_ix, all_cur_instance_iy] = find(instance == to_seg(i));
        all_cur_instance_pixel = sub2ind(size(label), all_cur_instance_ix, all_cur_instance_iy);
        
        objs{i}.instance = all_cur_instance_pixel;
        objs{i}.depth_map = ones(size(depth_map)) * max_depth;
        objs{i}.depth_map(objs{i}.instance) = depth_map(objs{i}.instance);
        
        objs{i}.old_pts = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, objs{i}.instance);
        objs{i}.new_pts = get_pt_on_new_coordinate_system(objs{i}.old_pts);
    end
end

function draw_scene(objs, index)
    figure(index)
    clf
    for i = 1 : length(objs)
        pts = objs{i}.new_pts;
        scatter3(pts(:,1), pts(:,2), pts(:,3), 3, 'r', 'fill')
        hold on
        draw_cuboid(objs{i}.cur_cuboid)
    end
    axis equal
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
function show_img_on_index(image, figure_ind)
    figure(figure_ind)
    imshow(image)
end
function params_cuboid_order = update_params(old_params, delta, gamma, activation_label)
    activation_label = (activation_label == 1);
    new_params = old_params;
    params_derivation_order = [new_params(3), new_params(1), new_params(2), new_params(4), new_params(5), new_params(6)];
    params_derivation_order(activation_label) = params_derivation_order(activation_label) + gamma * delta';
    params_cuboid_order = [params_derivation_order(2), params_derivation_order(3), params_derivation_order(1), params_derivation_order(4), params_derivation_order(5), params_derivation_order(6)];
    if params_cuboid_order(4) < 0 | params_cuboid_order(5) < 0 | params_cuboid_order(6) < 0
        params_cuboid_order = old_params;
        disp('Unstable')
    end
end
function activation_label = cancel_co_activation_label(activation_label)
    activation_label = (activation_label == 1);
    if (activation_label(2) | activation_label(3)) & (activation_label(5) | activation_label(4))
        if(randi([1 2], 1) == 1)
            activation_label(2) = 0; activation_label(3) = 0;
        else
            activation_label(5) = 0; activation_label(4) = 0;
        end
    end

end