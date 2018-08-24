% Edited by Sz on 08/18/18
function reconstructed_3d = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
    height = size(depth_map, 1);
    width = size(depth_map, 2);
    x = 1 : height; y = 1 : width;
    [X, Y] = meshgrid(y, x);
    pts = [Y(:) X(:)];
    
    projects_pts = [pts(valuable_ind,2) .* depth_map(valuable_ind), pts(valuable_ind,1) .* depth_map(valuable_ind), depth_map(valuable_ind), ones(length(valuable_ind), 1)];

    reconstructed_3d = (inv(intrinsic_params * extrinsic_params) * projects_pts')';
    % projects_pts = [pts(:,2) .* linear_depth_map(:), pts(:,1) .* linear_depth_map(:), linear_depth_map(:), ones(length(pts(:,1)), 1)];
    % projected_pts_2 = zeros(size(projects_pts));
    %{
    pt_count = 0;
    for i = 1 : height
        for j = 1 : width
            pt_count = pt_count + 1;
            cur_depth = depth_map(i, j);
            pixel_2d = [j * cur_depth, i * cur_depth, cur_depth, 1];
            projected_pts_2(pt_count, :) = (inv(intrinsic_params * extrinsic_params)  * pixel_2d')';
        end
    end
    %}
    %{
    depth_map_reprojected = zeros(height, width);
    real_xy = (intrinsic_params * extrinsic_params * reconstructed_3d')';
    real_xy(:,1) = real_xy(:,1) ./ real_xy(:,3);
    real_xy(:,2) = real_xy(:,2) ./ real_xy(:,3);
    linear_ind = sub2ind(size(depth_map), round(real_xy(:,2)), round(real_xy(:,1)));
    depth_map_reprojected(linear_ind) = real_xy(:, 3);
    max(max(abs(depth_map_reprojected - depth_map)))
    %}
    %{
    for i = 1 : length(linear_depth_map)
        if linear_depth_map(i) ~= depth_map(pts(i,1), pts(i,2))
            disp('error')
            break;
        end
    end
    %}
    %{
    f = sym('f', 'real');
    cx = sym('cx', 'real');
    cy = sym('cy', 'real');
    P = [f 0 cx 0; 0 f cy 0; 0 0 1 0; 0 0 0 1];
    %}
end
%{
% In order to inspect whether choose correct image areas or not
% linear_ind1 = sub2ind([size(label) 3], car_ix, car_iy, ones(length(car_ix), 1) * 1);
% linear_ind2 = sub2ind([size(label) 3], car_ix, car_iy, ones(length(car_ix), 1) * 2);
% linear_ind3 = sub2ind([size(label) 3], car_ix, car_iy, ones(length(car_ix), 1) * 3);
% linear_ind = [linear_ind1;linear_ind2;linear_ind3];
%}
%{
    % Gather all points together
    num_tot_ground_points = 0;
    for i = 1 : size(reconstructed_3d, 1)
        num_tot_ground_points = num_tot_ground_points + size(reconstructed_3d{i}, 1);
    end
    tot_ground_points = zeros(num_tot_ground_points, 3);
    num_tot_ground_points = 0;
    for i = 1 : size(reconstructed_3d, 1)
        tot_ground_points(num_tot_ground_points + 1 : num_tot_ground_points + size(reconstructed_3d{i}, 1), :) = reconstructed_3d{i};
        num_tot_ground_points = num_tot_ground_points + size(reconstructed_3d{i}, 1);
    end
%}
%{
function affine_transformation = get_affine_transformation(origin, new_basis, param)
    pt_camera_origin_3d = origin;
    x_dir = new_basis(1, :);
    y_dir = new_basis(2, :);
    z_dir = new_basis(3, :);
    new_coord1 = [1 0 0];
    new_coord2 = [0 1 0];
    new_coord3 = [0 0 1];
    new_pts = [new_coord1; new_coord2; new_coord3];
    old_Coord1 = pt_camera_origin_3d + x_dir;
    old_Coord2 = pt_camera_origin_3d + y_dir;
    old_Coord3 = pt_camera_origin_3d + z_dir;
    old_pts = [old_Coord1; old_Coord2; old_Coord3];
    
    T_m = new_pts' * inv((old_pts - repmat(pt_camera_origin_3d, [3 1]))');
    transition_matrix = eye(4,4);
    transition_matrix(1:3, 1:3) = T_m;
    transition_matrix(1, 4) = -pt_camera_origin_3d * x_dir';
    transition_matrix(2, 4) = -pt_camera_origin_3d * y_dir';
    transition_matrix(3, 4) = -pt_camera_origin_3d * z_dir';
    affine_transformation = transition_matrix;
    % Check: 
    % (affine_transformation * [old_pts ones(3,1)]')'
end
%}
function [dir, origin] = get_affine_transformation_from_plane(param, pts)
    origin = mean(pts); origin = origin(1:3);
    dir1 = (rand_sample_pt_on_plane(param) - rand_sample_pt_on_plane(param)); dir1 = dir1 / norm(dir1);
    dir3 = param(1:3); dir3 = dir3 / norm(dir3);
    dir2 = cross(dir1, dir3); dir2 = dir2 / norm(dir2);
    dir =[dir1;dir2;dir3];
    % dir1_ = [1 0 0];
    % dir2_ = [0 1 0];
    % dir3_ = [0 0 1];
    % dir_ = [dir1_;dir2_;dir3_];
    % rotation_matrix = dir_' * inv(dir');
    % affine_transformation = get_affine_transformation(origin, dir, param);
    % Check:
    % affine_transformation = dir_' * inv(dir');
    % affine_transformation(4,4) = 1;
end
% Edited on 08/20/18
% Aimed to inspect Projection
function mean_error = check_projection(objs, extrinsic_params, intrinsic_params)
    load('affine_matrix.mat');
    
    extrinsic_params = extrinsic_params / affine_matrx;
    image_size = size(objs{1}.depth_map);
    
    sum_error = 0;
    tot_num = 0;
    for i = 1 : length(objs)
        reconstructed_3d_pts = objs{i}.new_pts;
        [projected, valid, dist, depth] = projectPoints(reconstructed_3d_pts, intrinsic_params(1:3, 1:3), extrinsic_params, [0,0,0,0,0], image_size, true);
        sum_error = sum_error + sum((depth - objs{i}.depth_map(objs{i}.instance)).^2);
        tot_num = tot_num + length(objs{i}.instance);
    end
    mean_error = sqrt(sum_error / tot_num);
end
function get_all_3d_pt(depth_map, extrinsic_params, intrinsic_params, label)
    load('type_color_map.mat')
    figure(1)
    clf
    for i = 1 : 15
        % Exclude void(0), reserved1(13), reserved2(14), sky(1),
        % road(3), sidewalk(4), fence(5), lanemarking(12)
        if i == 1 || i == 0 || i == 13 || i == 14 || i == 12 || i == 3 || i == 4 || i == 5
            continue
        end
        [ix, iy] = find(label == i);
        linear_ind = sub2ind(size(depth_map), ix, iy);
        old_pts = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, linear_ind);
        new_pts = get_pt_on_new_coordinate_system(old_pts);
        figure(1)
        scatter3(new_pts(:,1),new_pts(:,2),new_pts(:,3),3,type_color_map(i, :)/255,'fill')
        hold on
    end
    axis equal
end
function draw_segmented_objs(objs, rgb_seg)
    show_img = uint8(zeros(size(rgb_seg)));
    cmap = colormap;
    cmap = uint8(round(cmap * 255));
    for i = 1 : length(objs)
        color = cmap(randi([1 64]), :);
        if objs{i}.type == 2
            [I,J] = ind2sub(size(objs{i}.depth_map),objs{i}.linear_ind);
            for k = 1 : length(I)
                show_img(I(k), J(k), :) = color;
            end
        end
    end
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
    intern_dist = abs(pts(:,1:3)) - 1; intern_dist(intern_dist < 0) = 0;
    dist = sum(intern_dist.^2, 2); dist(dist == 0) = min(0.5 - abs(pts(dist == 0, 1 : 3)), [], 2);
    ave_dist = sum(dist) / size(pts, 1);
    % params = zeros(5, 4);
    % for i = 1 : 5
    %     params(i, :) = cuboid{i}.params;
    % end
    % dist = abs(pts * params') ./ repmat(sum(params.^2, 2)', [size(pts, 1) 1]);
    % [val, ~] = min(dist');
    % ave_dist = sum(val) / size(pts, 1);
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

function ave_dist = calculate_ave_distance(cuboid, pts)
    global tot_count;
    theta_ = -cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; bottom_center = mean(cuboid{5}.pts);
    % pts_cpy = pts;
    
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
    dist = sum(intern_dist.^2, 2); dist = dist.^0.5; dist(dist == 0) = min(0.5 - abs(pts(dist == 0, 1 : 3)), [], 2);
    ave_dist = sum(dist) / size(pts, 1);
    
    %{
    cmap = colormap;
    [val, ind] = sort(dist);
    linear_ind = 1 : size(dist);
    color_ind = ceil(linear_ind / (max(linear_ind) / 63));
    figure(4)
    clf
    draw_cuboid(cuboid)
    hold on
    scatter3(pts_cpy(ind,1),pts_cpy(ind,2),pts_cpy(ind,3),3,cmap(color_ind, :),'fill')
    view(-30, 37);
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Matlab_code/Synthia_3D_scenen_reconstruction/exp_re/';
    
    save([path num2str(tot_count) '.mat']);
    tot_count = tot_count + 1;
    %}
end

function [visible_pt_3d, visible_pt_2d, visible_depth] = find_visible_pt_global(objects, pts_2d, pts_3d, depth, cam_m, transition_m, cam_origin)
    M = inv(cam_m * transition_m);
    visible_pt = zeros(size(pts_2d, 1), 7);
    deviation_threshhold = 0.01;
    valid_plane_num = 4;
    num_obj = size(objects, 1);
    for ii = 1 : size(pts_2d, 1)
        single_pt_all_possible_pos = zeros(valid_plane_num * num_obj, 4);
        valid_label = false(valid_plane_num * num_obj, 1);
        for k = 1 : num_obj
            cuboid = objects{k};
            for i = 1 : valid_plane_num
                params = cuboid{i}.params;
                z = - params * M(:, 4) / (pts_2d(ii, 1) * params * M(:, 1) + pts_2d(ii, 2) * params * M(:, 2) + params * M(:, 3));
                single_pt_all_possible_pos((k - 1) * valid_plane_num + i, :) = (M * [pts_2d(ii, 1) * z pts_2d(ii, 2) * z z 1]')';
            end
            [valid_label((k-1) * valid_plane_num + 1 : k * valid_plane_num, :), ~] = judge_on_cuboid(cuboid, single_pt_all_possible_pos((k - 1) * valid_plane_num + 1 : k * valid_plane_num, :));
        end
        
        if length(single_pt_all_possible_pos(valid_label)) > 0
            vale_pts = single_pt_all_possible_pos(valid_label, :);
            dist_to_origin = sum((vale_pts(:, 1:3) - cam_origin).^2, 2);
            shortest_ind = find(dist_to_origin == min(dist_to_origin));
            shortest_ind = shortest_ind(1);
            if(sum((vale_pts(shortest_ind, 1:3) - pts_3d(ii, 1:3)).^2) < deviation_threshhold)
                visible_pt(ii, 1:3) = vale_pts(shortest_ind, 1:3);
                visible_pt(ii, 4:5) = pts_3d(ii, 5:6);
                visible_pt(ii, 6) = pts_3d(ii, 4);
                visible_pt(ii, 7) = 1;
            end
        end
        
    end
    visible_label = visible_pt(:, 7) == 1;
    visible_pt_3d = visible_pt(visible_label, 1:6);
    visible_pt_2d = pts_2d(visible_label, 1:2);
    visible_depth = depth(visible_label);
end

function [valid_label, type] = judge_on_cuboid(cuboid, pts)
    valid_label = false([length(pts) 1]);
    type = ones([length(pts) 1]) * (-1);
    th = 0.01;
    if size(pts, 2) == 3
        pts = [pts ones(length(pts), 1)];
    end
    for i = 1 : 4
        pts_local_coordinate = (cuboid{i}.T * pts')';
        jdg_re = (pts_local_coordinate(:, 1) > -th & pts_local_coordinate(:, 1) < cuboid{i}.length1 + th) & (pts_local_coordinate(:, 3) > 0 - th & pts_local_coordinate(:, 3) < cuboid{i}.length2 + th) & (abs(pts_local_coordinate(:, 2)) < th);
        valid_label = valid_label | jdg_re;
        type(jdg_re) = i;
    end
end

function [hessian, first_order, tot_diff_record] = analytical_gradient(cuboid, P, T, visible_pt_3d, depth_map, hessian, first_order, activation_label)
    theta = cuboid{1}.theta;
    l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2;
    center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    M = P * T;
    % 3D points
    depth_map = [depth_map depth_map(:, end)];
    depth_map = [depth_map; depth_map(end, :)];
    
    pts_3d = cell(1, 4);
    pts_3d{1} = @(theta, xc, yc, l, w, h, k1, k2)[
        xc - 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k1 * cos(theta) * l;
        yc - 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k1 * sin(theta) * l;
        k2 * h
        ];
    pts_3d{2} = @(theta, xc, yc, l, w, h, k1, k2)[
        xc + 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k1 * sin(theta);
        yc + 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k1 * cos(theta);
        k2 * h
        ];
    pts_3d{3} = @(theta, xc, yc, l, w, h, k1, k2)[
        xc + 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k1 * l * cos(theta);
        yc + 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k1 * l * sin(theta);
        k2 * h
        ];
    pts_3d{4} = @(theta, xc, yc, l, w, h, k1, k2)[
        xc - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k1 * sin(theta);
        yc - 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k1 * cos(theta);
        k2 * h
        ];
    % 3D points' gradient on theta
    gra_pts_3d_theta = cell(1, 4);
    gra_pts_3d_theta{1} = @(theta, xc, yc, l, w, h, k1, k2)[
        1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k1 * l * sin(theta);
        -1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k1 * l * cos(theta);
        0
        ];
    gra_pts_3d_theta{2} = @(theta, xc, yc, l, w, h, k1, k2)[
        -1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k1 * cos(theta);
        1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k1 * sin(theta);
        0
        ];
    gra_pts_3d_theta{3} = @(theta, xc, yc, l, w, h, k1, k2)[
        -1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k1 * l * sin(theta);
        1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k1 * l * cos(theta);
        0
        ];
    gra_pts_3d_theta{4} = @(theta, xc, yc, l, w, h, k1, k2)[
        1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k1 * cos(theta);
        - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k1 * sin(theta);
        0
        ];
    % 3D points' gradient on xc
    gra_pts_3d_xs = cell(1, 4);
    gra_pts_3d_xs{1} = @(theta, xc, yc, l, w, h, k1, k2)[
        1;
        0;
        0
        ];
    gra_pts_3d_xs{2} = @(theta, xc, yc, l, w, h, k1, k2)[
        1;
        0;
        0
        ];
    gra_pts_3d_xs{3} = @(theta, xc, yc, l, w, h, k1, k2)[
        1;
        0;
        0
        ];
    gra_pts_3d_xs{4} = @(theta, xc, yc, l, w, h, k1, k2)[
        1;
        0;
        0
        ];
    % 3D points' gradient on yc
    gra_pts_3d_ys = cell(1, 4);
    gra_pts_3d_ys{1} = @(theta, xc, yc, l, w, h, k1, k2)[
        0;
        1;
        0
        ];
    gra_pts_3d_ys{2} = @(theta, xc, yc, l, w, h, k1, k2)[
        0;
        1;
        0 
        ];
    gra_pts_3d_ys{3} = @(theta, xc, yc, l, w, h, k1, k2)[
        0;
        1;
        0 
        ];
    gra_pts_3d_ys{4} = @(theta, xc, yc, l, w, h, k1, k2)[
        0;
        1;
        0
        ];
    % 3D points' gradient on l
    gra_pts_3d_l = cell(1, 4);
    gra_pts_3d_l{1} = @(theta, xc, yc, l, w, h, k1, k2)[
        -1 / 2 * cos(theta) + k1 * cos(theta);
        -1 / 2 * sin(theta) + k1 * sin(theta);
        0
        ];
    gra_pts_3d_l{2} = @(theta, xc, yc, l, w, h, k1, k2)[
        1 / 2 * cos(theta);
        1 / 2 * sin(theta);
        0
        ];
    gra_pts_3d_l{3} = @(theta, xc, yc, l, w, h, k1, k2)[
        1 / 2 * cos(theta) - k1 * cos(theta);
        1 / 2 * sin(theta) - k1 * sin(theta);
        0
        ];
    gra_pts_3d_l{4} = @(theta, xc, yc, l, w, h, k1, k2)[
        - 1 / 2 * cos(theta);
        - 1 / 2 * sin(theta);
        0
        ];
    % 3D points' gradient on w
    gra_pts_3d_w = cell(1, 4);
    gra_pts_3d_w{1} = @(theta, xc, yc, l, w, h, k1, k2)[
        1 / 2 * sin(theta);
        - 1 / 2 * cos(theta);
        0
        ];
    gra_pts_3d_w{2} = @(theta, xc, yc, l, w, h, k1, k2)[
        1 / 2 * sin(theta) - k1 * sin(theta);
        -1 / 2 * cos(theta) + k1 * cos(theta);
        0
        ];
    gra_pts_3d_w{3} = @(theta, xc, yc, l, w, h, k1, k2)[
        -1 / 2 * sin(theta);
        1 / 2 * cos(theta);
        0
        ];
    gra_pts_3d_w{4} = @(theta, xc, yc, l, w, h, k1, k2)[
        -1 / 2 * sin(theta) + k1 * sin(theta);
        1 / 2 * cos(theta) - k1 * cos(theta);
        0;
        ];
    gra_pts_3d_h = cell(1, 4);
    gra_pts_3d_h{1} = @(theta, xc, yc, l, w, h, k1, k2)[
        0;
        0;
        k2
        ];
    gra_pts_3d_h{2} = @(theta, xc, yc, l, w, h, k1, k2)[
        0;
        0;
        k2
        ];
    gra_pts_3d_h{3} = @(theta, xc, yc, l, w, h, k1, k2)[
        0;
        0;
        k2
        ];
    gra_pts_3d_h{4} = @(theta, xc, yc, l, w, h, k1, k2)[
        0;
        0;
        k2
        ];
    gradient_set = cell(1, 6);
    gradient_set{1} = gra_pts_3d_theta;
    gradient_set{2} = gra_pts_3d_xs;
    gradient_set{3} = gra_pts_3d_ys;
    gradient_set{4} = gra_pts_3d_l;
    gradient_set{5} = gra_pts_3d_w;
    gradient_set{6} = gra_pts_3d_h;
       
    activation_label = (activation_label == 1);
    
    k1 = visible_pt_3d(:, 4); k2 = visible_pt_3d(:, 5);
    
    px_ = @(pt_affine_3d)round((M(1, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
    py_ = @(pt_affine_3d)round((M(2, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
    ground_truth_depth_ = @(px, py) depth_map(px, py);
    estimated_depth_ = @(pt_affine_3d) M(3, :) * pt_affine_3d';
    diff_ = @(pt_affine_3d) ground_truth_depth_(py_(pt_affine_3d), px_(pt_affine_3d)) - estimated_depth_(pt_affine_3d);
    Ix_ = @(px, py)depth_map(py, px + 1) - depth_map(py, px);
    Iy_ = @(px, py)depth_map(py + 1, px) - depth_map(py, px);
    gpx_ = @(pt_affine_3d) (M(1, :) * (M(3, :) * pt_affine_3d') - M(3, :) * (M(1, :) * pt_affine_3d')) / (M(3, :) * pt_affine_3d')^2;
    gpy_ = @(pt_affine_3d) (M(2, :) * (M(3, :) * pt_affine_3d') - M(3, :) * (M(2, :) * pt_affine_3d')) / (M(3, :) * pt_affine_3d')^2;
    
    tot_diff_record = 0;
    for i = 1 : length(k1)
        plane_ind = visible_pt_3d(i, 6);
        
        % Calculate Diff_val
        pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
        try
            diff = diff_(pt_affine_3d);
            tot_diff_record = tot_diff_record + diff^2;
        catch ME
            disp([num2str(i) ' skipped'])
            length(k1)
            continue;
        end
        % Calculate J3
        J_x = zeros(6, 4);
        for j = 1 : 6
           J_x(j, :) = ([gradient_set{j}{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 0])';
        end
        J_x = J_x(activation_label, :);
        J_3 = M(3, :) * J_x';
        
        % Calculate J2
        px = px_(pt_affine_3d);
        py = py_(pt_affine_3d);
        Ix = Ix_(px, py);
        Iy = Iy_(px, py);
        gpx = gpx_(pt_affine_3d);
        gpy = gpy_(pt_affine_3d);
        J_2 = Ix * gpx * J_x' + Iy * gpy * J_x';
        
        % J
        % if(act_label(i))
        %     J = J_3 - J_2;
        % else
        %     J_3 = 0;
        %     J = J_3 - J_2;
        % end
        J = J_3 - J_2;
        
        hessian = hessian + J' * J;
        first_order = first_order + diff * J';
    end
    %{
        for i = 1 : length(k1)
            plane_ind = visible_pt_3d(i, 6);
             % Calculate Diff_val
            pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            try
                diff = diff_(pt_affine_3d);
                tot_diff_record = tot_diff_record + diff^2;
            catch ME
                disp([num2str(i) ' skipped'])
                length(k1)
                continue;
            end
            
            J_x = zeros(6, 4);
            for j = 1 : 6
                J_x(j, :) = ([gradient_set{j}{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 0])';
            end
            J_x = J_x(activation_label, :);
            J_3 = M(3, :) * J_x';
            
            % Calculate J2
            px = px_(pt_affine_3d);
            py = py_(pt_affine_3d);
            Ix = Ix_(px, py);
            Iy = Iy_(px, py);
            gpx = gpx_(pt_affine_3d);
            gpy = gpy_(pt_affine_3d);
            J_2 = Ix * gpx * J_x' + Iy * gpy * J_x';
            
            J = J_3 - J_2;
            grad = grad + (diff - J * delta) * (-J);
        end
    %}
    %{
        slice = 100; delta_unit = delta / slice; square_value_record = zeros(200, 1);
        for kk = 1 : 200
            square_value = 0;
            p_delta = delta_unit * kk;
            for i = 1 : length(k1)
                plane_ind = visible_pt_3d(i, 6);
                % Calculate Diff_val
                pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
                try
                    diff =   diff_(pt_affine_3d);
                    tot_diff_record = tot_diff_record + diff^2;
                catch ME
                    disp([num2str(i) ' skipped'])
                    length(k1)
                    continue;
                end
                
                J_x = zeros(6, 4);
                for j = 1 : 6
                    J_x(j, :) = ([gradient_set{j}{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 0])';
                end
                J_x = J_x(activation_label, :);
                J_3 = M(3, :) * J_x';
                
                % Calculate J2
                px = px_(pt_affine_3d);
                py = py_(pt_affine_3d);
                Ix = Ix_(px, py);
                Iy = Iy_(px, py);
                gpx = gpx_(pt_affine_3d);
                gpy = gpy_(pt_affine_3d);
                J_2 = Ix * gpx * J_x' + Iy * gpy * J_x';
                J = J_3 - J_2;
                
                square_value = square_value + (diff - J * p_delta)^2;
            end
            square_value_record(kk) = square_value;
        end
        stem(square_value_record - mean(square_value_record))
        diff_record(ppp) = tot_diff_record;
        theta = theta + delta(1); xc = xc + delta(2); yc = yc + delta(3);
    %}
end

