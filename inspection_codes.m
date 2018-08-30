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
                    diff =  diff_(pt_affine_3d);
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
        figure(1)
        clf
        stem(square_value_record - mean(square_value_record))
        diff_record(ppp) = tot_diff_record;
        
        
        
        small_delta = 0.0001;
        analtical_diff = 0; sum_num1 = 0; sum_num2 = 0; sum_ana = 0;
        theta1 = theta + small_delta; xc1 = xc + small_delta * 0; yc1 = yc +  small_delta * 0;
        theta2 = theta - small_delta; xc2 = xc - small_delta * 0; yc2 = yc -  small_delta * 0;
        for i = 1 : length(k1)
            plane_ind = visible_pt_3d(i, 6);
            % Calculate Diff_val
            pt_affine_3d1 = [pts_3d{plane_ind}(theta1, xc1, yc1, l, w, h, k1(i), k2(i)); 1]';
            pt_affine_3d2 = [pts_3d{plane_ind}(theta2, xc2, yc2, l, w, h, k1(i), k2(i)); 1]';
            try
                diff1 =  diff_(pt_affine_3d1); diff1 = diff1^2;
                diff2 =  diff_(pt_affine_3d2); diff2 = diff2^2;
            catch ME
                disp([num2str(i) ' skipped'])
                length(k1)
                continue;
            end
            sum_num1 = sum_num1 + diff1; sum_num2 = sum_num2 + diff2;
        end
        numerical_diff = sum_num1 - sum_num2;
        
        activation_label = [1 0 0 0 0 0]; activation_label = (activation_label == 1);
        for i = 1 : length(k1)
            plane_ind = visible_pt_3d(i, 6);
            
            % Calculate Diff_val
            pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            try
                diff = diff_(pt_affine_3d);
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
            
            J = J_3 - J_2;
            
            sum_ana = sum_ana + 2 * diff * (-J);
        end
        sum_ana = sum_ana * 2 * small_delta;
        numerical_diff / sum_ana;
        
        
        theta = theta + delta(1); xc = xc + delta(2); yc = yc + delta(3);
%}

% Edited on 08/25/2018
function [hessian, first_order, tot_diff_record] = analytical_gradient(cuboid, P, T, visible_pt_3d, depth_map, hessian, first_order, activation_label)
    activation_label = [1 0 0 0 0 0];
    theta = cuboid{1}.theta + 1;
    l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2;
    center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    M = P * T;
    % 3D points
    depth_map = [depth_map depth_map(:, end)];
    depth_map = [depth_map; depth_map(end, :)];
    depth_map_cpy = depth_map; black_cpy = zeros(size(depth_map_cpy));
    diff_record = zeros(100, 1);
    for ppp = 1 : 100
        hessian = zeros(1,1);
        first_order = zeros(1,1);
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
        pts_affine_3d_record = zeros(length(k1), 4);
        cur_cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
        for i = 1 : length(k1)
            plane_ind = visible_pt_3d(i, 6);
            
            % Calculate Diff_val
            pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            pts_affine_3d_record(i, :) = pt_affine_3d;
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
            if depth_map(py, px) == 0
                continue
            end
            Ix = Ix_(px, py);
            Iy = Iy_(px, py);
            gpx = gpx_(pt_affine_3d);
            gpy = gpy_(pt_affine_3d);
            J_2 = Ix * gpx * J_x' + Iy * gpy * J_x';
            
            J = J_3 - J_2;
            
            hessian = hessian + J' * J;
            first_order = first_order + diff * J';
            
            depth_map_cpy(py, px) = estimated_depth_(pt_affine_3d);
            black_cpy(py, px) = estimated_depth_(pt_affine_3d)';
        end
        delta = inv(hessian) * first_order;
        theta = theta + delta;
        diff_record(ppp) = tot_diff_record;
        figure(1)
        clf
        draw_cuboid(cur_cuboid)
        hold on
        scatter3(pts_affine_3d_record(:,1),pts_affine_3d_record(:,2),pts_affine_3d_record(:,3),3,'r','fill')
        view(-55, 20);
        %{
        figure(2)
        imshow(uint16(depth_map_cpy * 100000));
        figure(3)
        imshow(uint16(black_cpy * 100000));
        
        % Do gradient Check
        delta_theta = 0.000001;
        theta1 = theta + delta_theta; theta2 = theta - delta_theta; tot_diff_record = 0;
        tot_diff_record1 = 0; tot_diff_record2 = 0; J_tot = 0;
        for i = 1 : length(k1)
            plane_ind = visible_pt_3d(i, 6);
            
            % Calculate Diff_val
            pt_affine_3d1 = [pts_3d{plane_ind}(theta1, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            pt_affine_3d2 = [pts_3d{plane_ind}(theta2, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            try
                diff1 = diff_(pt_affine_3d1); diff2 = diff_(pt_affine_3d2); diff = diff_(pt_affine_3d);
                tot_diff_record1 = tot_diff_record1 + diff1^2; tot_diff_record2 = tot_diff_record2 + diff2^2;
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
            
            J_2 = 0;
            J = J_3 - J_2;
            
            hessian = hessian + J' * J;
            first_order = first_order + diff * J';
            
            depth_map_cpy(py, px) = estimated_depth_(pt_affine_3d);
            black_cpy(py, px) = estimated_depth_(pt_affine_3d)';
            J_tot = J_tot - diff * 2 * J;
        end
        J_tot
        (tot_diff_record1 - tot_diff_record2) / 2 / delta_theta
        %}
        % Perceiving how the algorithm is directing the function to its
        % global minimum value
        
        % I need to first get its global minimum value
        % If only with a single parameter, I can give this function with
        % multiple values and select the one with the minimum
        delta_theta = theta -theta / 3000 : theta / 300000 : theta + theta / 3000; diff_record = zeros(length(delta_theta), 1);
        acc_j_record = zeros(length(delta_theta), 1);
        for p = 1 : length(delta_theta)
            theta = delta_theta(p);
            tot_diff = 0;
            acc_j = 0;
            for i = 1 : length(k1)
                plane_ind = visible_pt_3d(i, 6);
                
                % Calculate Diff_val
                pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
                try
                    diff = diff_(pt_affine_3d);
                    tot_diff = tot_diff + diff^2;
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
                
                J_2 = 0;
                J = J_3 - J_2;
                acc_j = acc_j + 2 * J * diff;
                
                hessian = hessian + J' * J;
                first_order = first_order + diff * J';
            end
            diff_record(p) = tot_diff;
            acc_j_record(p) = acc_j;
        end
        figure(2)
        stem(diff_record - mean(diff_record))
        (diff_record(20) - diff_record(19)) / (delta_theta(20) - delta_theta(19))
        acc_j_record(20)
    end
end

theta_range = -2 * pi : 0.01 : 2 * pi; loss_record = zeros(size(theta_range, 1), 1);
tot_img_val_record = zeros(length(theta_range), 1); estimated_depth_record = zeros(length(theta_range), 1);
for m = 1 : length(theta_range)
    cur_theta = theta_range(m); tot_diff_record = 0; tot_img_val = 0; tot_estimated_depth = 0;
    for i = 1 : length(k1)
        plane_ind = visible_pt_3d(i, 6);
        pt_affine_3d = [pts_3d{plane_ind}(cur_theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
        try
            px = px_(pt_affine_3d); py = py_(pt_affine_3d);
            gt_depth = ground_truth_depth_(py, px); tot_img_val = tot_img_val + gt_depth;
            diff = diff_(pt_affine_3d);
            tot_diff_record = tot_diff_record + diff^2;
            tot_estimated_depth = tot_estimated_depth + estimated_depth_(pt_affine_3d);
        catch ME
            disp([num2str(i) ' skipped'])
            continue;
        end
    end
    tot_img_val_record(m) = tot_img_val;
    loss_record(m) = tot_diff_record;
    estimated_depth_record(m) = tot_estimated_depth;
end
figure(1)
stem(estimated_depth_record,'filled')
figure(2)
clf;
stem(theta_range, loss_record, 'filled')

function [hessian, first_order, tot_diff_record] = analytical_gradient(cuboid, P, T, visible_pt_3d, depth_map, hessian, first_order, activation_label)
    activation_label = [1 0 0 0 0 0]; activation_label = (activation_label == 1);
    theta = cuboid{1}.theta + 1;
    l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2;
    center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    M = P * T;
    % 3D points
    depth_map = [depth_map depth_map(:, end)];
    depth_map = [depth_map; depth_map(end, :)];
    depth_map_cpy = depth_map; black_cpy = zeros(size(depth_map_cpy));
    diff_record = zeros(100, 1);
    
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
    gradient_set{1} = gra_pts_3d_theta; gradient_set{2} = gra_pts_3d_xs; gradient_set{3} = gra_pts_3d_ys;
    gradient_set{4} = gra_pts_3d_l; gradient_set{5} = gra_pts_3d_w; gradient_set{6} = gra_pts_3d_h;
    
    px_ = @(pt_affine_3d)round((M(1, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d')); py_ = @(pt_affine_3d)round((M(2, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
    ground_truth_depth_ = @(px, py) depth_map(px, py);
    estimated_depth_ = @(pt_affine_3d) M(3, :) * pt_affine_3d';
    diff_ = @(pt_affine_3d) ground_truth_depth_(py_(pt_affine_3d), px_(pt_affine_3d)) - estimated_depth_(pt_affine_3d);
    Ix_ = @(px, py)depth_map(py, px + 1) - depth_map(py, px); Iy_ = @(px, py)depth_map(py + 1, px) - depth_map(py, px);
    gpx_ = @(pt_affine_3d) (M(1, :) * (M(3, :) * pt_affine_3d') - M(3, :) * (M(1, :) * pt_affine_3d')) / (M(3, :) * pt_affine_3d')^2;
    gpy_ = @(pt_affine_3d) (M(2, :) * (M(3, :) * pt_affine_3d') - M(3, :) * (M(2, :) * pt_affine_3d')) / (M(3, :) * pt_affine_3d')^2;
    
    k1 = visible_pt_3d(:, 4); k2 = visible_pt_3d(:, 5);
    for ppp = 1 : 100
        hessian = zeros(1,1); first_order = zeros(1,1);
        tot_diff_record = 0; pts_affine_3d_record = zeros(length(k1), 4);
        cur_cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
        for i = 1 : 1
            plane_ind = visible_pt_3d(i, 6);
            % Calculate Diff_val
            pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            pts_affine_3d_record(i, :) = pt_affine_3d;
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
            if depth_map(py, px) == 0
                continue
            end
            Ix = Ix_(px, py);
            Iy = Iy_(px, py);
            gpx = gpx_(pt_affine_3d);
            gpy = gpy_(pt_affine_3d);
            J_2 = Ix * gpx * J_x' + Iy * gpy * J_x';
            
            J = J_3 - J_2;
            
            hessian = hessian + J' * J;
            first_order = first_order + diff * J';
        end
        delta = inv(hessian) * first_order; theta = theta + delta;
        
        %{
        delta_theta = theta -theta / 30 : theta / 3000 : theta + theta / 30; diff_record = zeros(length(delta_theta), 1);
        acc_j_record = zeros(length(delta_theta), 1);
        for p = 1 : length(delta_theta)
            local_theta = delta_theta(p);
            tot_diff = 0; acc_j = 0;
            for i = 1 : length(k1)
                plane_ind = visible_pt_3d(i, 6);
                % Calculate Diff_val
                pt_affine_3d = [pts_3d{plane_ind}(local_theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
                try
                    diff = diff_(pt_affine_3d);
                    tot_diff = tot_diff + diff^2;
                catch ME
                    disp([num2str(i) ' skipped'])
                    length(k1)
                    continue;
                end
                % Calculate J3
                J_x = zeros(6, 4);
                for j = 1 : 6
                    J_x(j, :) = ([gradient_set{j}{plane_ind}(local_theta, xc, yc, l, w, h, k1(i), k2(i)); 0])';
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
                acc_j = acc_j + 2 * J * diff;
                
                hessian = hessian + J' * J;
                first_order = first_order + diff * J';
            end
            diff_record(p) = tot_diff;
            acc_j_record(p) = acc_j;
        end
        figure(2)
        stem( -theta / 30 : theta / 3000 : theta / 30, diff_record - mean(diff_record))
        
        theta = theta + delta;
        %}
    end
end
function [hessian, first_order, tot_diff_record] = analytical_gradient(cuboid, P, T, visible_pt_3d, depth_map, hessian, first_order, activation_label)
    theta = cuboid{1}.theta;
    l = cuboid{1}.length1 + 1; w = cuboid{2}.length1; h = cuboid{1}.length2;
    center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    M = P * T;
    % 3D points
    depth_map = [depth_map depth_map(:, end)];
    depth_map = [depth_map; depth_map(end, :)];
    
    activation_label = [0 0 0 1 0 0]; activation_label = (activation_label == 1); activated_params_num = 1;
    hessian = zeros(activated_params_num, activated_params_num); first_order = zeros(activated_params_num, 1);
    
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
    % diff_ = @(pt_affine_3d) ground_truth_depth_(py_(pt_affine_3d), px_(pt_affine_3d)) - estimated_depth_(pt_affine_3d);
    diff_ = @(pt_affine_3d) 3 - estimated_depth_(pt_affine_3d);
    Ix_ = @(px, py)depth_map(py, px + 1) - depth_map(py, px);
    Iy_ = @(px, py)depth_map(py + 1, px) - depth_map(py, px);
    gpx_ = @(pt_affine_3d) (M(1, :) * (M(3, :) * pt_affine_3d') - M(3, :) * (M(1, :) * pt_affine_3d')) / (M(3, :) * pt_affine_3d')^2;
    gpy_ = @(pt_affine_3d) (M(2, :) * (M(3, :) * pt_affine_3d') - M(3, :) * (M(2, :) * pt_affine_3d')) / (M(3, :) * pt_affine_3d')^2;
    
    diff_record = zeros(100, 1); delta_record = zeros(100, 1); pts_affine_3d_record = zeros(100, 3);
    estimated_depth_record = zeros(100,1); ground_truth_depth_record = zeros(100, 1);
    ix_record = zeros(100,1); iy_record = zeros(100,1); J_record = zeros(100,1); sign_diff_record = zeros(100 ,1); l_record = zeros(100, 1);
    for it_num = 1 : 3500
        tot_diff_record = 0; ground_truth = 0; ix = 0; iy = 0; first_order = 0; hessian = 0;
        for i = 1 : 1
            plane_ind = visible_pt_3d(i, 6);
            
            % Calculate Diff_val
            pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            try
                diff = diff_(pt_affine_3d);
                tot_diff_record = tot_diff_record + diff^2; estimated_depth = estimated_depth_(pt_affine_3d);
                % ix = px_(pt_affine_3d); iy = py_(pt_affine_3d);
                % ground_truth = ground_truth_depth_(iy, ix);
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
            % Ix = Ix_(px, py);
            % Iy = Iy_(px, py);
            % gpx = gpx_(pt_affine_3d);
            % gpy = gpy_(pt_affine_3d);
            % J_2 = Ix * gpx * J_x' + Iy * gpy * J_x';
            % J_2 = 0;
            
            % J
            % if(act_label(i))
            %     J = J_3 - J_2;
            % else
            %     J_3 = 0;
            %     J = J_3 - J_2;
            % end
            % J = J_3 - J_2;
            J = J_3;
            
            hessian = hessian + J' * J;
            first_order = first_order + diff * J';
        end
        %{
        delta_test = 0.000001;
        l1 = l + delta_test; l2 = l - delta_test;
        pt_affine_3d1 = [pts_3d{plane_ind}(theta, xc, yc, l1, w, h, k1(i), k2(i)); 1]';
        estimated_depth1 = estimated_depth_(pt_affine_3d1);
        pt_affine_3d2 = [pts_3d{plane_ind}(theta, xc, yc, l2, w, h, k1(i), k2(i)); 1]';
        estimated_depth2 = estimated_depth_(pt_affine_3d2);
        diff1 = diff_(pt_affine_3d1); diff2 = diff_(pt_affine_3d2);
        diff = diff_(pt_affine_3d);
        grad = (diff1^2 - diff2^2) / 2 / delta_test;
        % grad = (estimated_depth1 - estimated_depth2) / 2 / delta_test
        grad_ = - 2 * J * diff;
        if abs(grad - grad_) > 0.000001
            error('gradient wrong')
        end
        %}
        lastwarn('')
        delta = inv(hessian) * first_order;
        [warnMsg, ~] = lastwarn;
        if ~isempty(warnMsg)
            a = 1;
        end
        if it_num == 81 || it_num == 178
            a = 1;
        end
        % theta = theta + 0.001 * delta;
        l = l + 0.1 * delta;
        diff_record(it_num) = tot_diff_record; delta_record(it_num) = delta; pts_affine_3d_record(it_num, :) = pt_affine_3d(1:3);
        estimated_depth_record(it_num) = estimated_depth; ground_truth_depth_record(it_num) = ground_truth; J_record(it_num) = J;
        sign_diff_record(it_num) = diff; l_record(it_num) = l - 0.1 * delta;
        % ix_record(it_num) = ix; iy_record(it_num) = iy;
    end
    figure(1); clf; stem(diff_record,'filled', 'Marker', '.');
    figure(2); clf; stem(delta_record, 'Marker', '.');
    figure(3); clf; scatter3(pts_affine_3d_record(:,1),pts_affine_3d_record(:,2),pts_affine_3d_record(:,3),3,'r','filled')
    figure(4); clf; stem(estimated_depth_record);
    figure(5); clf; stem(ground_truth_depth_record);
    figure(7); clf; stem(J_record, 'Marker', '.'); figure(8); clf; stem(sign_diff_record, 'Marker', '.');
    figure(9); clf; stem(l_record, 'Marker', '.');
    % figure(6); clf; stem(ix_record); figure(7); clf; stem(iy_record);
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
