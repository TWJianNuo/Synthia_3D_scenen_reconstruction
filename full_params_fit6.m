sim_env()
function sim_env()
    close all;
    x = 10;
    y = 10;
    theta = pi / 2;
    l = 4;
    w = 6;
    h = 8;
    gamma = 0.1;
    max_it_num = 300;
    path = '/Users/zhushengjie/Downloads/sim_env_data_sequence/';
    init_activation_label = [1 1 1 1 1 0];
    num1 = 10; num2 = 10;
    diff_threshold = 0.1;
    isblocked = true;
    
    y_track = -10: 3 : 50;
    x_track = 12 * ones(1, length(y_track));
    pt_camera_origin_3d_stack = [x_track', y_track', ones(length(x_track), 1) * 3];
    pt_camera_origin_3d = pt_camera_origin_3d_stack(1, :);
    
    x_dir = [1 0 0];
    x_dir = x_dir / norm(x_dir);
    y_dir_before = [0 0 -1];
    y_dir = y_dir_before - y_dir_before * x_dir' * x_dir;
    y_dir = y_dir / norm(y_dir);
    z_dir = cross(x_dir, y_dir);
    new_basis = [x_dir; y_dir; z_dir];
    [~, T] = get_Cam_and_Affine_matrix(pt_camera_origin_3d, new_basis);
    P = [
        100,   0     150   0;
        0,     100,  150   0;
        0,     0     1     0;
        0,     0     0     1;
        ];
    image_size = [300 300];
    % added_img = generate_circle_on_rectangle_img(image_size, [150 150], 50);
    % figure(1)
    % show_depth_map(added_img);

    [objects, params] = generate_cuboids_in_the_sim_scene();
    guess_params = params;
    T_cluster = cell(length(y_track), 1);
    depth_map_cluster = cell(length(y_track), 1);
    seg_map_cluster = cell(length(y_track), 1);
    % view_depth_map(params, pt_camera_origin_3d_stack, P, T, image_size, new_basis)
    % generate_data_sequence(objects, pt_camera_origin_3d_stack, P, T, image_size, new_basis);
    for i = 1 : 1 : 12
        pt_camera_origin_3d = pt_camera_origin_3d_stack(i, :);
        [~, T] = get_Cam_and_Affine_matrix(pt_camera_origin_3d, new_basis);
        [depth_map, seg_map] = provide_depth_and_seg_info(path, i);
        figure(3)
        clf
        show_depth_map(depth_map)
        T_cluster{i} = T;
        depth_map_cluster{i} = depth_map;
        seg_map_cluster{i} = seg_map;
        if i == 1
            guess_params = init_guess_for_cuboid(depth_map, seg_map, P , T, init_activation_label);
        end
        if i > 11
            guess_params = find_local_optimized_cuboid_for_all(objects, P, T_cluster, pt_camera_origin_3d_stack, guess_params, gamma, init_activation_label, 10, max_it_num, diff_threshold, depth_map_cluster, image_size, seg_map_cluster, isblocked, pt_camera_origin_3d);
        end
    end
    figure(1)
    draw_scene(1, objects, pt_camera_origin_3d_stack, pt_camera_origin_3d, new_basis, 10, guess_params, depth_map, seg_map, image_size, path, i);
    per_exp_time = 100;
    single_param_converge_exp = zeros(6, 100);
    dev_threshhold = 0.1;
    ground_truth_params = [x,y,theta,l,w,h];
end
function added_img = generate_circle_on_rectangle_img(image_size, center_point, max_depth_val)
    added_img = zeros(image_size(2), image_size(1));
    x = center_point(2);
    y = center_point(1);
    for i = 1 : image_size(2)
        for j = 1 : image_size(1)
            added_img(i, j) = sqrt((i - x)^2 + (j - y)^2) * max_depth_val / 300;
        end
    end
end
function view_depth_map(params, pt_camera_origin_3d_stack, P, T, image_size, new_basis)
    pt_camera_origin_3d = pt_camera_origin_3d_stack(1, :);
    [~, T] = get_Cam_and_Affine_matrix(pt_camera_origin_3d, new_basis);
    P = [
        100,   0     150   0;
        0,     100,  150   0;
        0,     0     1     0;
        0,     0     0     1;
        ];
    params = params(1,:);
    for i = 1 : 10 : 100
        cur_params = params;
        cur_params(3) = cur_params(3) + i / 180 * pi;
        objects{1} = generate_cuboid_by_center(cur_params(1), cur_params(2), cur_params(3), cur_params(4), cur_params(5), cur_params(6));
        [depth_map, seqmentation_map, visible_pt_3d] = acqurie_2d_ground_truth_image(objects, pt_camera_origin_3d, P, T, image_size);
        figure(1)
        clf
        show_depth_map(depth_map);
        figure(2)
        clf
        draw_cubic_shape_frame(objects{1})
        hold on
        scatter3(visible_pt_3d(:,1),visible_pt_3d(:,2), visible_pt_3d(:,3),'r','.')
        axis equal
    end
end
function guess_params = init_guess_for_cuboid(depth_map, seg_map, P , T, init_activation_label)
    num_objects = max(max(seg_map));
    guess_params = zeros(num_objects, 6);
    for i = 1 : num_objects
        [indx, indy] = find(seg_map == i);
        depth = depth_map(seg_map == i);
        pts_3d_after_reconstruction = (inv(P * T) * [indy .* depth indx .* depth, depth, ones(length(indx), 1)]')';
        [params, cuboid] = estimate_rectangular(pts_3d_after_reconstruction);
        % params(4) = params(4) + 1;
        % params(5) = params(5) + 1;
        params = acquire_random_init_params(params(1), params(2), params(3), params(4), params(5), params(6), init_activation_label);
        guess_params(i, :) = params;
    end
end
function guess_cuboids_set = generate_cuboids_by_params(params)
    guess_cuboids_set = cell(size(params, 1), 1);
    for i = 1 : size(params, 1)
        guess_cuboids_set{i} = generate_cuboid_by_center(params(i, 1),params(i, 2),params(i, 3),params(i, 4),params(i, 5),params(i, 6));
    end
end
function params = find_local_optimized_cuboid_for_all(objects, P, T_cluster, pt_camera_origin_3d_stack, params, gamma, activation_label, smaple_num, max_it_num, diff_threshold, depth_map_cluster, image_size, seg_map_cluster, isblocked, origin)
    guess_cuboids_set = generate_cuboids_by_params(params);
    for i = 1 : size(objects, 1)
        cuboid = objects{i};
        current_params = params(i, :);
        params(i, :) = find_local_optimized_cuboid(cuboid, P, T_cluster, pt_camera_origin_3d_stack, current_params, gamma, activation_label, smaple_num, max_it_num, diff_threshold, depth_map_cluster, image_size, i, seg_map_cluster, isblocked, guess_cuboids_set, origin);
        figure(3)
        saveas(gcf,[num2str(i) '_re.png'])
    end
end

function current_params = find_local_optimized_cuboid(cuboid, P, T_cluster, pt_camera_origin_3d_stack, current_params, gamma, activation_label, smaple_num, max_it_num, diff_threshold, depth_map_cluster, image_size, obj_ind, seg_map_cluster, isblocked, objects, origin)
    num1 = smaple_num;
    num2 = smaple_num;
    it_num = 0;
    it_max_num = max_it_num;
    delta = 0.02;
    threshold = 0.1;
    delta_threshold = 0.01;
    sum_diff_threshold = 0.01;
    colors = [
        [0.961009523809524,0.889019047619048,0.153676190476191];
        [0.279466666666667,0.344671428571429,0.971676190476191];
        [0.246957142857143,0.791795238095238,0.556742857142857];
        [0.989200000000000,0.813566666666667,0.188533333333333];
        ];
    changes_container = zeros(5, 6);
    % current_params(1: 5) = current_params(1 : 5) + rand([1 5]) .* current_params(1 : 5) / 10;
    while(it_num < it_max_num)
        it_num = it_num + 1;
        cur_activation_label = cancel_co_activation_label(activation_label);
        guess_cubic = generate_cuboid_by_center(current_params(1), current_params(2), current_params(3), current_params(4), current_params(5), current_params(6));
        objects{obj_ind} = guess_cubic;
        pts_estimated_3d = sample_cubic_by_num(guess_cubic, num1, num2);
        non_empty_ind = find(~cellfun('isempty', T_cluster));
        it_count = 0;
        for q = 3 : 3 % length(non_empty_ind)
            it_count = it_count + 1;
            T = T_cluster{non_empty_ind(q)};
            seg_map = seg_map_cluster{non_empty_ind(q)};
            depth_map = depth_map_cluster{non_empty_ind(q)};
            [iyy, ixx] = find(seg_map == obj_ind);
            meanx = mean(ixx); meany = mean(iyy);
            if isnan(meanx)
                break;
            end
            added_img = generate_circle_on_rectangle_img(image_size, [meanx meany], 500);
            slector = seg_map ~= obj_ind;
            depth_map(slector) = depth_map(slector) + added_img(slector);
            % depth_map(slector) = added_img(slector);
            pt_camera_origin_3d = pt_camera_origin_3d_stack(non_empty_ind(q), :);
            [pts_estimated_2d, pts_estimated_vlaid, ~, depth] = projectPoints(pts_estimated_3d, P(1:3, :), T, [0,0,0,0,0], [image_size(1) image_size(2)], false);
            pts_estimated_3d = pts_estimated_3d(pts_estimated_vlaid, :);
            pts_estimated_2d = pts_estimated_2d(pts_estimated_vlaid, :);
            depth = depth(pts_estimated_vlaid);
            if sum(int8(pts_estimated_vlaid)) == 0
                disp('No points get projected onto the plane')
                break;
            end
            [visible_pt_3d, visible_pt_2d, visible_depth] = find_visible_pt_global(objects, pts_estimated_2d, pts_estimated_3d, depth, P, T, pt_camera_origin_3d);
            act_label = true(size(visible_depth, 1));
            if isblocked
                try
                    ix = floor(visible_pt_2d(:, 1)) + 1;
                    iy = floor(visible_pt_2d(:, 2)) + 1;
                    % act_label = (ix<=0 | ix > image_size(2) | iy<=0 | iy > image_size(1));
                    % ix(act_label) = [];
                    % iy(act_label) = [];
                    linearInd = sub2ind(size(seg_map), iy, ix);
                    act_label = (seg_map(linearInd) == obj_ind);
                    % pts_estimated_vlaid(~act_label) = pts_estimated_vlaid(~act_label) & (seg_map(linearInd) == obj_ind);
                    % pts_estimated_vlaid(act_label) = false;
                catch ME
                    disp('Discreterization error caused discarded')
                end
            end
            if it_count == 1
                activated_params_num = sum(int8(cur_activation_label));
                hessian = zeros(activated_params_num, activated_params_num);
                first_order = zeros(activated_params_num, 1);
                sum_diff = 0;
                num = 0;
            end
            [hessian, first_order, sum_diff, num, J_diff_record_1, J_diff_record_2, J_diff_record_sum] = analytical_gradient(guess_cubic, P, T, visible_pt_3d, depth_map, cur_activation_label, threshold, hessian, first_order, sum_diff, num, act_label);
        end
        sum_diff = sum_diff / (num + 0.00001);
        delta = inv(hessian) * first_order;
        if max(abs(delta)) < delta_threshold || max(abs(sum_diff)) < sum_diff_threshold || it_num > max_it_num
            break;
        end
        changes_container(mod(it_num - 1, size(changes_container, 1)) + 1, cur_activation_label) = delta;
        if it_num > size(changes_container, 1) && max(abs(mean(changes_container))) < 0.01
            break;
        end
        [J_diff_record_color_map, indices] = grab_cmap_color(J_diff_record_1);
        figure(3)
        clf
        draw_cubic_shape_frame_with_color(guess_cubic, colors(obj_ind, :));
        hold on
        draw_cuboid(cuboid);
        hold on;
%         scatter3(visible_pt_3d(:,1),visible_pt_3d(:,2),visible_pt_3d(:,3),15,J_diff_record_color_map(:,:),'fill')
%         
%         [J_diff_record_color_map, indices] = grab_cmap_color(J_diff_record_2);
%         figure(4)
%         clf
%         draw_cubic_shape_frame_with_color(guess_cubic, colors(obj_ind, :));
%         hold on
%         draw_cuboid(cuboid);
%         hold on;
%         scatter3(visible_pt_3d(act_label,1),visible_pt_3d(act_label,2),visible_pt_3d(act_label,3),15,J_diff_record_color_map(act_label, :),'fill')
%         
%         [J_diff_record_color_map, indices] = grab_cmap_color(J_diff_record_sum);
%         figure(5)
%         clf
%         draw_cubic_shape_frame_with_color(guess_cubic, colors(obj_ind, :));
%         hold on
%         draw_cuboid(cuboid);
%         hold on;
%         scatter3(visible_pt_3d(:,1),visible_pt_3d(:,2),visible_pt_3d(:,3),15,J_diff_record_color_map,'fill')
        
        current_params = update_params(current_params, delta, cur_activation_label, gamma);
    end
end
function [J_diff_record_color_map, indices] = grab_cmap_color(J_diff_record)
    cmap = colormap;
    min_val = min(J_diff_record);
    max_val = max(J_diff_record);
    indices = floor((J_diff_record - min_val) / max_val * (size(cmap, 1)-1)) + 1;
    J_diff_record_color_map = cmap(indices, :);
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

function guess_params = get_random_init_guess_for_each_cubic(params, activation_label)
    guess_params = params;
    for i = 1 : size(params, 1)
        x = params(i, 1); y = params(i, 2); theta = params(i, 3); l = params(i, 4); w = params(i, 5); h = params(i, 6);
        guess_params(i, :) = acquire_random_init_params(x, y, theta, l, w, h, activation_label);
    end
end

function [depth_map, seqmentation_map] = provide_depth_and_seg_info(path, time_frame)
    load([path 'frame_' num2str(time_frame) 'data.mat']);
end


function generate_data_sequence(objects, pt_camera_origin_3d_stack, P, T, image_size, new_basis)
    path = '/Users/zhushengjie/Downloads/sim_env_data_sequence/';
    for i = 1 : length(pt_camera_origin_3d_stack)
        pt_camera_origin_3d = pt_camera_origin_3d_stack(i, :);
        [~, T] = get_Cam_and_Affine_matrix(pt_camera_origin_3d, new_basis);
        P = [
            100,   0     150   0;
            0,     100,  150   0;
            0,     0     1     0;
            0,     0     0     1;
            ];
        [depth_map, seqmentation_map] = acqurie_2d_ground_truth_image(objects, pt_camera_origin_3d, P, T, image_size);
        figure(2);
        clf
        show_depth_map(depth_map)
        saveas(gcf, [path 'depth_' num2str(i) '.png']);
        figure(3);
        clf
        show_seg_map(seqmentation_map, 3, image_size)
        saveas(gcf, [path 'seg_' num2str(i) '.png']);
        save([path 'frame_' num2str(i) 'data'], 'depth_map', 'seqmentation_map');
    end
end

function [depth_map, seg_map, visible_pt_3d] = acqurie_2d_ground_truth_image(objects, origin, P, T, image_size)
    max_depth = 50;
    height = image_size(2); width = image_size(1);
    x = 1 : height;
    y = 1 : width;
    [xx, yy] = meshgrid(x, y);
    pts_2d = [xx(:), yy(:)];
    depth_map = zeros(height, width);
    seg_map = zeros(height, width);
    [visible_pt_3d, visible_depth, visible_pt_2d, visible_label, visbile_pt_seg_label] = find_intersected_3dpts_with_cuboid(objects, pts_2d, P, T, origin);
    for i = 1 : size(visible_pt_2d, 1)
        depth_map(visible_pt_2d(i, 2), visible_pt_2d(i, 1)) = visible_depth(i);
        seg_map(visible_pt_2d(i, 2), visible_pt_2d(i, 1)) = visbile_pt_seg_label(i);
    end
    [image_pts, reconstructed_3d, ground_sky_label] = get_ground_depth(pts_2d, visible_label, P, T, max_depth);
    
    for i = 1 : size(image_pts, 1)
        depth_map(image_pts(i, 2), image_pts(i, 1)) = image_pts(i, 3);
        seg_map(image_pts(i, 2), image_pts(i, 1)) = ground_sky_label(i);
    end
    % show_seg_map(seg_map, 2, image_size);
    % figure(2)
    % show_depth_map(depth_map)
end

function image = generate_seg_map(seg_map, image_size)
    load('color.mat')
    height = image_size(2); width = image_size(1);
    image=zeros(height,width,3);
    num_obj = max(max(seg_map));
    for i = 1 : size(seg_map, 1)
        for j = 1 : size(seg_map, 2)
            image(i, j, :) = rgb_val(seg_map(i, j) + 3, :);
        end
    end
end

function show_seg_map(seg_map, figure_index, image_size)
    figure(figure_index);
    load('color.mat')
    height = image_size(2); width = image_size(1);
    image=zeros(height,width,3);
    num_obj = max(max(seg_map));
    for i = 1 : size(seg_map, 1)
        for j = 1 : size(seg_map, 2)
            image(i, j, :) = rgb_val(seg_map(i, j) + 3, :);
        end
    end
    imshow(uint8(image));
end

function [objects, params] = generate_cuboids_in_the_sim_scene()
    x = 12;
    y = 12;
    theta = 0;
    l = 8;
    w = 4;
    h = 12;
    
    num_objects = 4;
    objects = cell(num_objects, 1);
    params = zeros(num_objects, 6);
    
    objects{1} = generate_cuboid_by_center(x + 20, y, theta, l, w, h);
    objects{2} = generate_cuboid_by_center(x - 20, y, theta, l, w, h);
    objects{3} = generate_cuboid_by_center(x + 20, y + 15, theta, l, w, h);
    objects{4} = generate_cuboid_by_center(x - 20, y + 15, theta, l, w, h);
    
    params(1, :) = [x + 20, y, theta, l, w, h];
    params(2, :) = [x - 20, y, theta, l, w, h];
    params(3, :) = [x + 20, y + 15, theta, l, w, h];
    params(4, :) = [x - 20, y + 15, theta, l, w, h];
end
function draw_scene(index_of_figure, objects, pt_camera_origin_3d_stack, pt_camera_origin_3d, new_basis, length, params, depth_map, segmentaion_map, image_size, path, time_sequence)
    figure(index_of_figure);
    clf;
    for i = 1 : size(objects,1)
        hold on;
        draw_cuboid(objects{i});
    end
    hold on
    draw_image_plane(pt_camera_origin_3d, new_basis, length, length);
    hold on
    plot3(pt_camera_origin_3d_stack(:,1), pt_camera_origin_3d_stack(:,2), pt_camera_origin_3d_stack(:,3), 'r.')
    for i = 1 : size(params,1)
        hold on;
        cuboid = generate_cuboid_by_center(params(i,1), params(i,2), params(i,3), params(i,4), params(i,5), params(i,6));
        draw_cubic_shape_frame(cuboid)
    end
    view(-10, 45)
    axis equal
    saveas(gcf,[path 'scene_' num2str(time_sequence) '.png']);
    figure(2)
    clf;
    segmentaion_map = generate_seg_map(segmentaion_map, image_size);
    depth_map = generate_depth_map_img(depth_map);
    rgbImage = [cat(3, depth_map, depth_map, depth_map) segmentaion_map];
    imshow(rgbImage)
    saveas(gcf,[path 'depth_and_seg_' num2str(time_sequence) '.png']);
end
function [image_pts, reconstructed_3d, ground_sky_label] = get_ground_depth(pts_2d, visible_label, P, T, max_depth)
    params = [0 0 1 0];
    M = inv(P * T);
    pts_2d = pts_2d(~visible_label, :);
    ground_sky_label = zeros(sum(visible_label == 0), 1);
    reconstructed_3d = zeros(size(pts_2d, 1), 4);
    ground_depth = zeros(size(pts_2d, 1), 1);
    for i = 1 : size(pts_2d, 1)
        z = - params * M(:, 4) / (pts_2d(i, 1) * params * M(:, 1) + pts_2d(i, 2) * params * M(:, 2) + params * M(:, 3));
        reconstructed_3d(i, :) = (M * [pts_2d(i, 1) * z pts_2d(i, 2) * z z 1]')';
        ground_depth(i) = z;
    end
    image_pts = [pts_2d ground_depth];
    image_pts(ground_depth > max_depth | ground_depth <= 0, 3) = max_depth;
    ground_sky_label(~(ground_depth <= 0)) = -1;
    ground_sky_label(ground_depth <= 0) = -2;
end
function depth_map = generate_depth_map_img(depth_map)
    depth_map = uint8(depth_map / max(max(depth_map)) * 255);
end
function show_depth_map(depth_map)
    depth_map = uint8(depth_map / max(max(depth_map)) * 255);
    imshow(depth_map);
end
function show_color_depth_map(depth_map)
    color_depth_map = zeros(size(depth_map,1),size(depth_map,2),3);
    cmap = colormap;
    max_depth = max(max(depth_map));
    ind = floor(depth_map / max_depth * length(cmap));
    for i = 1 : size(depth_map,1)
        for j = 1 : size(depth_map, 2)
            color_depth_map(i, j, :) = cmap(ind(i, j), :);
        end
    end
    imshow(color_depth_map)
end

function [visible_pt_3d, visible_depth, visible_pt_2d, visible_label, segmentation] = find_intersected_3dpts_with_cuboid(objects, pts_2d, cam_m, transition_m, cam_origin)
    M = inv(cam_m * transition_m);
    visible_pt_3d = zeros(size(pts_2d, 1), 3);
    visible_pt_2d = zeros(size(pts_2d, 1), 2);
    visible_depth = zeros(size(pts_2d, 1), 1);
    segmentation = zeros(size(pts_2d, 1), 1);
    valid_plane_num = 4;
    for ii = 1 : size(pts_2d, 1)
        single_pt_all_possible_pos = zeros(valid_plane_num * size(objects, 1), 5);
        depth_record = zeros(valid_plane_num * size(objects, 1), 1);
        valid_label = false(valid_plane_num * size(objects, 1), 1);
        for t = 1 : size(objects, 1)
            cuboid = objects{t};
            for i = 1 : valid_plane_num
                params = cuboid{i}.params;
                z = - params * M(:, 4) / (pts_2d(ii, 1) * params * M(:, 1) + pts_2d(ii, 2) * params * M(:, 2) + params * M(:, 3));
                single_pt_all_possible_pos((t-1) * valid_plane_num + i, 1:4) = (M * [pts_2d(ii, 1) * z pts_2d(ii, 2) * z z 1]')';
                single_pt_all_possible_pos((t-1) * valid_plane_num + i, 5) = t;
                depth_record((t-1) * valid_plane_num + i) = z;
                if min(z) < 0
                    a = 1;
                end
            end
        end
        for t = 1 : size(objects, 1)
            cuboid = objects{t};
            [cur_valid_label, plane_ind] = judge_on_cuboid(cuboid, single_pt_all_possible_pos((t-1) * valid_plane_num + 1 : t * valid_plane_num, 1 : 4));
            cur_valid_label = cur_valid_label & depth_record((t-1) * valid_plane_num + 1 :  t * valid_plane_num) > 0;
            valid_label((t-1) * valid_plane_num + 1 : t * valid_plane_num) = cur_valid_label;
        end
        if length(single_pt_all_possible_pos(valid_label)) > 0
            vale_pts = single_pt_all_possible_pos(valid_label, 1:5);
            vale_depth_record = depth_record(valid_label);
            dist_to_origin = sum((vale_pts(:, 1:3) - cam_origin).^2, 2);
            shortest_ind = find(dist_to_origin == min(dist_to_origin));
            shortest_ind = shortest_ind(1);
            visible_pt_3d(ii, :) = vale_pts(shortest_ind, 1:3);
            visible_depth(ii, :) = vale_depth_record(shortest_ind);
            visible_pt_2d(ii, :) = pts_2d(ii, :);
            segmentation(ii) = vale_pts(shortest_ind, 5);
        end
    end
    selector = (visible_depth ~= 0);
    visible_pt_3d = visible_pt_3d(selector, :);
    visible_depth = visible_depth(selector);
    visible_pt_2d = visible_pt_2d(selector, :);
    segmentation = segmentation(selector);
    visible_label = selector;
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

function activation_label = acquire_random_activation_label(init_activation_label)
    theta_mark = init_activation_label(1);
    x_mark = init_activation_label(2);
    y_mark = init_activation_label(3);
    l_w_h_mark = init_activation_label(4) | init_activation_label(5) | init_activation_label(6);
    rand_max_num = double(theta_mark) + double(x_mark) + double(y_mark) + double(l_w_h_mark);
    k = randi([1 rand_max_num], 1);
    
    selector = [theta_mark, x_mark, y_mark, l_w_h_mark];
    selector = (selector == 1);
    t = false(1, rand_max_num); t(k) = true;
    
    fin_mark = zeros(4, 1);
    fin_mark(selector', :) = fin_mark(selector', :) + double(t)';
    fin_mark = (fin_mark == 1);
    
    activation_label = false(1, 6);
    if fin_mark(1)
        activation_label = [1 0 0 0 0 0];
    end
    if fin_mark(2)
        activation_label = [0 1 0 0 0 0];
    end
    if fin_mark(3)
        activation_label = [0 0 1 0 0 0];
    end
    if fin_mark(4)
        activation_label = [0 0 0 1 1 1];
        activation_label = activation_label & init_activation_label;
    end
end
function params = acquire_random_init_params(x, y, theta, l, w, h, activation_label)
    deviations = [(rand(1) - 0.5) * x * 0.1; (rand(1) - 0.5) * y * 0.1; (rand(1) - 0.5) * 2 * pi * 0.1; (rand(1) + 1) * l * 0.05; (rand(1) + 1) * w * 0.05; (rand(1) + 1) * h * 0.05];
    right_order_activation_label = activation_label;
    right_order_activation_label(3) = activation_label(1);
    right_order_activation_label(1) = activation_label(2);
    right_order_activation_label(2) = activation_label(3);
    deviations = deviations' .* double(right_order_activation_label);
    % x = x + (rand(1) - 0.5) * x * 0.3;
    % y = y + (rand(1) - 0.5) * y * 0.3;
    % theta = theta + (rand(1) - 0.5) * theta * 0.3;
    % l = l + (rand(1) - 1) * l * 0.2;
    % w = w + (rand(1) - 1) * w * 0.2;
    params = [x, y, theta, l, w, h] + deviations;
end
function [ground_truth_depth, ground_truth_3d] = grab_ground_truth_depth(cuboid, pts_2d, pts_3d, depth, cam_m, transition_m, cam_origin)
    M = inv(cam_m * transition_m);
    ground_truth_depth = zeros(size(pts_2d, 1), 1); % 5 for valid 6 for type
    ground_truth_3d = zeros(size(pts_2d, 1), 3);
    valid_plane_num = 4;
    for ii = 1 : size(pts_2d, 1)
        depth_record = zeros(valid_plane_num, 1);
        single_pt_all_possible_pos = zeros(valid_plane_num, 4);
        for i = 1 : valid_plane_num
            params = cuboid{i}.params;
            z = - params * M(:, 4) / (pts_2d(ii, 1) * params * M(:, 1) + pts_2d(ii, 2) * params * M(:, 2) + params * M(:, 3));
            single_pt_all_possible_pos(i, :) = (M * [pts_2d(ii, 1) * z pts_2d(ii, 2) * z z 1]')';
            depth_record(i) = z;
        end
        [valid_label, plane_ind] = judge_on_cuboid(cuboid, single_pt_all_possible_pos);
        if length(single_pt_all_possible_pos(valid_label)) > 0
            vale_pts = single_pt_all_possible_pos(valid_label, :);
            depth_record = depth_record(valid_label);
            dist_to_origin = sum((vale_pts(:, 1:3) - cam_origin).^2, 2);
            shortest_ind = find(dist_to_origin == min(dist_to_origin));
            shortest_ind = shortest_ind(1);
            ground_truth_depth(ii, 1) = depth_record(shortest_ind);
            ground_truth_3d(ii, :) = vale_pts(shortest_ind, 1:3);
        end
    end
    ground_truth_3d = ground_truth_3d(ground_truth_depth ~= 0, :);
end
function [visible_pt_3d, visible_pt_2d, visible_depth] = find_visible_pt(cuboid, pts_2d, pts_3d, depth, cam_m, transition_m, cam_origin)
    M = inv(cam_m * transition_m);
    visible_pt = zeros(size(pts_2d, 1), 7);
    deviation_threshhold = 0.01;
    valid_plane_num = 4;
    for ii = 1 : size(pts_2d, 1)
        single_pt_all_possible_pos = zeros(valid_plane_num, 4);
        for i = 1 : valid_plane_num
            params = cuboid{i}.params;
            z = - params * M(:, 4) / (pts_2d(ii, 1) * params * M(:, 1) + pts_2d(ii, 2) * params * M(:, 2) + params * M(:, 3));
            single_pt_all_possible_pos(i, :) = (M * [pts_2d(ii, 1) * z pts_2d(ii, 2) * z z 1]')';
        end
        [valid_label, plane_ind] = judge_on_cuboid(cuboid, single_pt_all_possible_pos);
        if length(single_pt_all_possible_pos(valid_label)) > 0
            vale_pts = single_pt_all_possible_pos(valid_label, :);
            type_val = plane_ind(valid_label);
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
function pts = sample_cubic_by_num(cuboid, num1, num2)
    max = 0.9;
    min = 0.1;
    k1 = min : (max - min)/(num1 - 1) : max;
    k2 = min : (max - min)/(num2 - 1) : max;
    pts = zeros(4 * num1 * num2, 6); % 3D location plus belongings, last two for k1 and k2
    for i = 1 : 4
        x = k1 * cuboid{i}.length1 * cuboid{i}.plane_dirs(1, 1) + cuboid{i}.pts(1, 1);
        y = k1 * cuboid{i}.length1 * cuboid{i}.plane_dirs(1, 2) + cuboid{i}.pts(1, 2);
        z = k2 * cuboid{i}.length2;        
        [xx, zz] = meshgrid(x, z);
        [yy, zz] = meshgrid(y, z);
        [kk1, kk2] = meshgrid(k1, k2);
        pts((i - 1) * num1 * num2 + 1 : i * num1 * num2, :) = [xx(:) yy(:) zz(:) ones(length(zz(:)), 1) * i kk1(:) kk2(:)];
    end
end
function draw_image_plane(origin, new_basis, length, width)
    pt_camera_origin_3d = origin;
    x_dir = new_basis(1, :);
    y_dir = new_basis(2, :);
    z_dir = new_basis(3, :);
    mesh_pt1 = pt_camera_origin_3d - y_dir * width - x_dir * length;
    mesh_pt2 = pt_camera_origin_3d - y_dir * width + x_dir * length;
    mesh_pt3 = pt_camera_origin_3d + y_dir * width + x_dir * length;
    mesh_pt4 = pt_camera_origin_3d + y_dir * width - x_dir * length;
    mesh_pts = [mesh_pt1;mesh_pt2;mesh_pt3;mesh_pt4];
    hold on
    scatter3(pt_camera_origin_3d(1), pt_camera_origin_3d(2), pt_camera_origin_3d(3), 3, 'g')
    hold on
    patch('Faces',[1 2 3 4],'Vertices',mesh_pts,'FaceAlpha', 0.5)
end
function [P, T] = get_Cam_and_Affine_matrix(origin, new_basis)
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
    
    % Transition matrix
    T_m = new_pts' * inv((old_pts - repmat(pt_camera_origin_3d, [3 1]))');
    transition_matrix = eye(4,4);
    transition_matrix(1:3, 1:3) = T_m;
    transition_matrix(1, 4) = -pt_camera_origin_3d * x_dir';
    transition_matrix(2, 4) = -pt_camera_origin_3d * y_dir';
    transition_matrix(3, 4) = -pt_camera_origin_3d * z_dir';
    T = transition_matrix;
    
    % Camera matrix
    cam_m = [
        10,   0   200   0;
        0,   10,  200   0;
        0,   0   1   0;
        0,   0   0   1;
        ];
    P = cam_m;
end


function [hessian, first_order, sum_diff, num, J_diff_record_1, J_diff_record_2, J_diff_record_sum] = analytical_gradient(cuboid, P, T, visible_pt_3d, depth_map, activation_label, threshold, hessian, first_order, sum_diff, num, act_label)
    theta = cuboid{1}.theta;
    l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2;
    center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    params = [theta, xc, yc, l, w, h, 1, 1];
    % delta = 0.00001;
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
    activated_params_num = sum(int8(activation_label));
    
    
    k1 = 0.1; k2 = 0.1;
    delta = 0.00001;
    for i = 1 : 6
        for j = 1 : 4
            params1 = params;
            params1(i) = params1(i) + delta;
            params2 = params;
            params2(i) = params2(i) - delta;
            grad_eqn = gradient_set{i}{j};
            theoretical_gradient = M(3, :) * [grad_eqn(params(1), params(2), params(3), params(4), params(5), params(6), k1, k2); 0];
            val1 = M(3, :) * [pts_3d{j}(params1(1), params1(2), params1(3), params1(4), params1(5), params1(6), k1, k2); 1];
            val2 = M(3, :) * [pts_3d{j}(params2(1), params2(2), params2(3), params2(4), params2(5), params2(6), k1, k2); 1];
            numerical_gradient = (val1 - val2) / 2 / delta;
            max(abs(theoretical_gradient - numerical_gradient));
            if (max(abs(theoretical_gradient - numerical_gradient)) > 0.00001)
                disp(['Error on [' num2str(i) ', ' num2str(j) ']'])
            end
        end
    end
    
    k1 = visible_pt_3d(:, 4); k2 = visible_pt_3d(:, 5);
    % hessian = zeros(activated_params_num, activated_params_num);
    % first_order = zeros(activated_params_num, 1);
    
    px_ = @(pt_affine_3d)round((M(1, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
    py_ = @(pt_affine_3d)round((M(2, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
    ground_truth_depth_ = @(px, py) depth_map(px, py);
    estimated_depth_ = @(pt_affine_3d) M(3, :) * pt_affine_3d';
    diff_ = @(pt_affine_3d) ground_truth_depth_(py_(pt_affine_3d), px_(pt_affine_3d)) - estimated_depth_(pt_affine_3d);
    Ix_ = @(px, py)depth_map(py, px + 1) - depth_map(py, px);
    Iy_ = @(px, py)depth_map(py + 1, px) - depth_map(py, px);
    gpx_ = @(pt_affine_3d) (M(1, :) * (M(3, :) * pt_affine_3d') - M(3, :) * (M(1, :) * pt_affine_3d')) / (M(3, :) * pt_affine_3d')^2;
    gpy_ = @(pt_affine_3d) (M(2, :) * (M(3, :) * pt_affine_3d') - M(3, :) * (M(2, :) * pt_affine_3d')) / (M(3, :) * pt_affine_3d')^2;
    J_diff_record_1 = zeros(length(k1), 1);
    J_diff_record_2 = zeros(length(k1), 1);
    J_diff_record_sum = zeros(length(k1), 1);
    
    % sum_diff = 0;
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
        sum_diff = sum_diff + abs(diff);
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
        if(act_label(i))
            J = J_3 - J_2;
            J_diff_record_2(i) = sum(sum(abs(J_3)));
        else
            J_3 = 0;
            J = J_3 - J_2;
            J_diff_record_2(i) = sum(sum(abs(J_3)));
        end
        
        J_diff_record_1(i) = sum(sum(abs(J_2)));
        J_diff_record_sum(i) = sum(sum(abs(J)));
        % J_record(i,1:3) = pt_affine_3d(1:3);
        % J_record(i,4) = J_3(1);
        % J_record(i,5) = J_3(2);
        
        %{
        delta = 0.00001;
        params = [theta, xc, yc, l, w, h, k1(i), k2(i)];
        px__ = @(pt_affine_3d)((M(1, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
        py__ = @(pt_affine_3d)((M(2, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
        for ii = 1 : 6
            params1 = params;
            params1(ii) = params1(ii) + delta;
            params2 = params;
            params2(ii) = params2(ii) - delta;
            pt_affine_3d1 = [pts_3d{plane_ind}(params1(1),params1(2),params1(3),params1(4),params1(5),params1(6),params1(7),params1(8)); 1]';
            pt_affine_3d2 = [pts_3d{plane_ind}(params2(1),params2(2),params2(3),params2(4),params2(5),params2(6),params2(7),params2(8)); 1]';
            px__1 = px__(pt_affine_3d1);
            px__2 = px__(pt_affine_3d2);
            jpx = (px__1 - px__2) / 2 /delta;
            jpx_comp = gpx * J_x';
            abs(jpx - jpx_comp(ii))
            
            py__1 = py__(pt_affine_3d1);
            py__2 = py__(pt_affine_3d2);
            jpy = (py__1 - py__2) / 2 /delta;
            jpy_comp = gpy * J_x';
            abs(jpy - jpy_comp(ii))
        end
        %}
        
        
        hessian = hessian + J' * J;
        first_order = first_order + diff * J';
        if isnan(hessian)
            a = 1;
        end
    end
    num = num + length(k1);
end

function params_cuboid_order = update_params(old_params, delta, activation_label, gamma)
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
function cuboid = generate_cuboid(x, y, theta, l, w, h)
    base_dir = [1 0 0];
    R = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
    R_90 = [0 -1 0; 1 0 0; 0 0 1];
    base_dirs = zeros(4, 3);
    base_dirs(1, :) = (R * base_dir')';
    base_dirs(2, :) = (R_90 * base_dirs(1, :)')';
    base_dirs(3, :) = (R_90 * base_dirs(2, :)')';
    base_dirs(4, :) = (R_90 * base_dirs(3, :)')';
    plane_dirs = [base_dirs(2, :);base_dirs(3, :); base_dirs(4, :); base_dirs(1, :)];
    
    cuboid = cell([6 1]);
    length = [l w l w];
    cuboid{1}.theta = theta;
    for i = 1 : 4
        cuboid{i}.pts = zeros(4, 3);
        if i == 1
            cuboid{i}.pts(1, :) = [x y 0];
        else
            cuboid{i}.pts(1, :) = cuboid{i - 1}.pts(2, :);
        end
        cuboid{i}.pts(2, :) = cuboid{i}.pts(1, :) + base_dirs(i, :) * length(i);
        cuboid{i}.pts(3, :) = cuboid{i}.pts(2, :) + [0 0 h];
        cuboid{i}.pts(4, :) = cuboid{i}.pts(1, :) + [0 0 h];
        cuboid{i}.dir = plane_dirs(i, :);
        cuboid{i}.length1 = length(i);
        cuboid{i}.length2 = h;
        cuboid{i}.params = [cuboid{i}.dir -(cuboid{i}.dir * cuboid{i}.pts(1, :)')];
        cuboid{i}.plane_dirs = [base_dirs(i, :); [0 0 1]];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dir1 = cuboid{i}.plane_dirs(1, :);
        dir2 = cuboid{i}.dir;
        dir3 = [0 0 1];
        new_dirs = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]';
        old_dirs = [[dir1 0]; [dir2 0]; [dir3 0]; [0 0 0 1]];
        cuboid{i}.T = new_dirs' * inv(old_dirs');
        Transition_matrix = -[cuboid{i}.pts(1, :) * dir1' cuboid{i}.pts(1, :) * dir2' cuboid{i}.pts(1, :) * dir3'];
        cuboid{i}.T(1:3, 4) = Transition_matrix';
    end
    i = 5;
    if i >= 5
        cuboid{i}.pts = [
            cuboid{1}.pts(1, :) + [0 0 h];
            cuboid{2}.pts(1, :) + [0 0 h];
            cuboid{3}.pts(1, :) + [0 0 h];
            cuboid{4}.pts(1, :) + [0 0 h];
            ];
        cuboid{i}.dir = [0 0 -1];
        cuboid{i}.length1 = l;
        cuboid{i}.length2 = w;
        cuboid{i}.params = [cuboid{i}.dir -(cuboid{i}.dir * cuboid{i}.pts(1, :)')];
        cuboid{i}.plane_dirs = [base_dirs(1, :); base_dirs(2, :)];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dir1 = cuboid{i}.plane_dirs(1, :);
        dir2 = cuboid{i}.plane_dirs(2, :);
        dir3 = [0 0 -1];
        new_dirs = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]';
        old_dirs = [[dir1 0]; [dir2 0]; [dir3 0]; [0 0 0 1]];
        cuboid{i}.T = new_dirs' * inv(old_dirs');
        Transition_matrix = -[cuboid{i}.pts(1, :) * dir1' cuboid{i}.pts(1, :) * dir2' cuboid{i}.pts(1, :) * dir3'];
        cuboid{i}.T(1:3, 4) = Transition_matrix';
    end
end
function draw_cuboid(cuboid)
    for i = 1 : 5
        hold on;
        patch('Faces',[1 2 3 4],'Vertices',cuboid{i}.pts,'FaceAlpha', 0.5)
    end
    xlabel('x');
    ylabel('y');
    zlabel('z');
    grid on
    axis equal
end
function draw_cubic_shape_frame(cuboid)
    col = 'r';
    pts1 = cuboid{1}.pts;
    pts2 = cuboid{3}.pts;
    cornerpoints = [pts1; pts2];
    line_width = 2;
    hx = cornerpoints(:,1);hy = cornerpoints(:,2);hz = cornerpoints(:,3);
    
    x=[hx(1);hx(2)];y=[hy(1);hy(2)];z=[hz(1);hz(2)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(2);hx(3)];y=[hy(2);hy(3)];z=[hz(2);hz(3)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(3);hx(4)];y=[hy(3);hy(4)];z=[hz(3);hz(4)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(4);hx(1)];y=[hy(4);hy(1)];z=[hz(4);hz(1)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(5);hx(6)];y=[hy(5);hy(6)];z=[hz(5);hz(6)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(6);hx(7)];y=[hy(6);hy(7)];z=[hz(6);hz(7)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(7);hx(8)];y=[hy(7);hy(8)];z=[hz(7);hz(8)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(8);hx(5)];y=[hy(8);hy(5)];z=[hz(8);hz(5)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(1);hx(6)];y=[hy(1);hy(6)];z=[hz(1);hz(6)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(2);hx(5)];y=[hy(2);hy(5)];z=[hz(2);hz(5)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(3);hx(8)];y=[hy(3);hy(8)];z=[hz(3);hz(8)];plot3(x,y,z,col,'LineWidth',line_width);hold on;
    x=[hx(4);hx(7)];y=[hy(4);hy(7)];z=[hz(4);hz(7)];plot3(x,y,z,col,'LineWidth',line_width);hold off;
end
function draw_cubic_shape_frame_with_color(cuboid, color)
    col = color;
    pts1 = cuboid{1}.pts;
    pts2 = cuboid{3}.pts;
    cornerpoints = [pts1; pts2];
    line_width = 2;
    hx = cornerpoints(:,1);hy = cornerpoints(:,2);hz = cornerpoints(:,3);
    
    x=[hx(1);hx(2)];y=[hy(1);hy(2)];z=[hz(1);hz(2)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(2);hx(3)];y=[hy(2);hy(3)];z=[hz(2);hz(3)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(3);hx(4)];y=[hy(3);hy(4)];z=[hz(3);hz(4)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(4);hx(1)];y=[hy(4);hy(1)];z=[hz(4);hz(1)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(5);hx(6)];y=[hy(5);hy(6)];z=[hz(5);hz(6)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(6);hx(7)];y=[hy(6);hy(7)];z=[hz(6);hz(7)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(7);hx(8)];y=[hy(7);hy(8)];z=[hz(7);hz(8)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(8);hx(5)];y=[hy(8);hy(5)];z=[hz(8);hz(5)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(1);hx(6)];y=[hy(1);hy(6)];z=[hz(1);hz(6)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(2);hx(5)];y=[hy(2);hy(5)];z=[hz(2);hz(5)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(3);hx(8)];y=[hy(3);hy(8)];z=[hz(3);hz(8)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold on;
    x=[hx(4);hx(7)];y=[hy(4);hy(7)];z=[hz(4);hz(7)];plot3(x,y,z,'color',col,'LineWidth',line_width);hold off;
end
function cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h)
    base_dir = [1 0 0];
    R = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
    R_90 = [0 -1 0; 1 0 0; 0 0 1];
    base_dirs = zeros(4, 3);
    base_dirs(1, :) = (R * base_dir')';
    base_dirs(2, :) = (R_90 * base_dirs(1, :)')';
    base_dirs(3, :) = (R_90 * base_dirs(2, :)')';
    base_dirs(4, :) = (R_90 * base_dirs(3, :)')';
    m = [cx, cy, 0] - base_dirs(1, :) * l / 2 - base_dirs(2, :) *w / 2;
    x = m(1);
    y = m(2);
    cuboid = generate_cuboid(x, y, theta, l, w, h);
end