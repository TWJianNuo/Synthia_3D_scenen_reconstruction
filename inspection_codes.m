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