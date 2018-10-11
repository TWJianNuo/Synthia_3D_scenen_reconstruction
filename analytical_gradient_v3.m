function params_end = analytical_gradient_v3(cuboid, intrinsic_param, extrinsic_param, depth_map, lin_ind, visible_pt_3d, init_cuboid)
    % load('supplementary_data/26_Sep_2018_15debug_ana_3.mat')
    global exp_num path;
    params = generate_cubic_params(cuboid); params_gt = params;
    params = mutate_params(params); params_initial = params;
    activation_label = [1 1 1 1 1 0]; activation_label = (activation_label == 1); it_num = 200; diff_record = zeros(it_num, 1);
    delta_record = zeros(it_num, 1); 
    if exp_num == 1 
        path = make_dir(); 
    end
    % clear; load('/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/27_Sep_2018_19/44.mat');
    for i = 1 : it_num
        delta_theta = get_delta_value(visible_pt_3d, params, extrinsic_param, intrinsic_param, activation_label, depth_map);
        diff = calculate_diff(depth_map, intrinsic_param, extrinsic_param, visible_pt_3d, params);
        diff_record(i) = diff;
        if(judge_stop(delta_theta, params, diff_record))
            break
        end
         % delta_record(i) = delta_theta;
        old_params = params; params = update_param(params, delta_theta, activation_label); 
        if mod(i, 200) == 1 | mod(i, 200) == 50 | mod(i, 200) == 100 | mod(i, 200) == 150 | mod(i, 200) == 200
            img1 = plot_scene(cuboid, params, visible_pt_3d);
            img2 = visualize_on_depth_map(depth_map, params, visible_pt_3d, extrinsic_param, intrinsic_param);
            save_img(path, i, img1, img2, exp_num)
        end
    end
    params_end = old_params;
    img1 = plot_scene(cuboid, params_end, visible_pt_3d);
    img2 = visualize_on_depth_map(depth_map, params_end, visible_pt_3d, extrinsic_param, intrinsic_param);
    save_img(path, i, img1, img2, exp_num)
    plot_and_save_stem(diff_record, path, exp_num)
    record_metric(params_gt, params_initial, params_end, diff_record, path, exp_num);
    exp_num = exp_num + 1;
end
function record_metric(params_gt, params_initial, params_end, diff_record, path, exp_num)
    diff_record = diff_record(diff_record~=0); numerical_stable_fac = 0.0000000000001;
    params_ratio = abs(params_gt - params_end + numerical_stable_fac) ./ abs(params_gt - params_initial + numerical_stable_fac);
    if exp_num == 1
        f1 = fopen([path '/' 'params_diff.txt'],'w');
        f2 = fopen([path '/' 'delta_record.txt'],'w');
        f3 = fopen([path '/' 'gt_params_record.txt'],'w');
        f4 = fopen([path '/' 'final_params_record.txt'],'w');
        print_matrix(f1, params_ratio);
        print_matrix(f2, [diff_record(1) diff_record(end)])
        print_matrix(f3, params_gt);
        print_matrix(f4, params_end);
    else
        f1 = fopen([path '/' 'params_diff.txt'],'a');
        f2 = fopen([path '/' 'delta_record.txt'],'a');
        f3 = fopen([path '/' 'gt_params_record.txt'],'a');
        f4 = fopen([path '/' 'final_params_record.txt'],'a');
        print_matrix(f1, params_ratio);
        print_matrix(f2, [diff_record(1) diff_record(end)])
        print_matrix(f3, params_gt);
        print_matrix(f4, params_end);
    end
    fclose(f1); fclose(f2);
end
function print_matrix(fileID, m)
    for i = 1 : size(m,1)
        for j = 1 : size(m,2)
            fprintf(fileID, '%d\t', m(i,j));
        end
        fprintf(fileID, '\n');
    end
end
function is_stop = judge_stop(delta, params, diff_record)
    delta_th = 0.5; params_th = 0.1; is_stop = false; diff_record = diff_record(diff_record~=0);
    if sum(abs(delta)) < delta_th
        is_stop = true;
    end
    if sum((params(4:5))) < params_th
        is_stop = true;
    end
    if length(diff_record) > 1
        if diff_record(end) > diff_record(end-1)
            is_stop = true;
        end
    end
end
function params = mutate_params(params)
    params(1) = params(2) + rand * 0.5;
    params(4) = params(4) + abs(rand) * 2;
    params(5) = params(5) + abs(rand) * 2;
    params(2) = params(2) + abs(rand) * 5;
    params(3) = params(3) + abs(rand) * 5;
end
function plot_and_save_stem(diff_record, path, exp_num)
    f = figure('visible', 'off'); stem(diff_record(diff_record~=0), 'fill'); F = getframe(f); [X, Map] = frame2im(F);
    imwrite(X, [path '/' num2str(exp_num) '_' 'error' '.png']);
end
function X = plot_scene(old_cuboid, params, visible_pt_3d)
    % figure(1); clf; scatter3(pts_new(:, 1), pts_new(:, 2), pts_new(:, 3), 3, 'g', 'fill'); hold on;
    f = figure('visible', 'off'); clf;
    cuboid = generate_cuboid_by_center(params(2), params(3), params(1), params(4), params(5), params(6)); draw_cubic_shape_frame(cuboid); sampled_pts = sample_cubic_by_num(cuboid, 10, 10);
    selector = (sampled_pts(:, 4) == visible_pt_3d(1, 3) | sampled_pts(:, 4) == visible_pt_3d(end, 3));
    sampled_pts = sampled_pts(selector, :);
    hold on; scatter3(sampled_pts(:,1), sampled_pts(:,2), sampled_pts(:,3), 3, 'r', 'fill')
    hold on; draw_cuboid(old_cuboid); F = getframe(f); [X, Map] = frame2im(F);
end
function save_workspace(base_dir, exp_num)
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path = [base_dir num2str(exp_num) '.mat']; save(path);
end
function X = visualize_on_depth_map(depth_map, params, visible_pt_3d, extrinsic_param, intrinsic_param)
    f = figure('visible', 'off'); clf;
    cuboid = generate_cuboid_by_center(params(2), params(3), params(1), params(4), params(5), params(6));
    sampled_pts = sample_cubic_by_num(cuboid, 10, 10);
    selector = (sampled_pts(:, 4) == visible_pt_3d(1, 3) | sampled_pts(:, 4) == visible_pt_3d(end, 3));
    sampled_pts = sampled_pts(selector, 1:3); [pts2d, depth] = project_point_2d(extrinsic_param, intrinsic_param, sampled_pts);
    depth_map = depth_map / max(max(depth_map));
    depth_map = cat(3, depth_map, depth_map, depth_map); depth_map = insertMarker(depth_map, pts2d, '*', 'color', 'red', 'size', 1);
    imshow(depth_map);
    F = getframe(f); [X, Map] = frame2im(F);
    % depth_map = add_depth_to_depth_map(depth_map, pts2d, 2000);
end
function save_img(path, frame, img1, img2, exp_num)
    imwrite(img1, [path '/' num2str(exp_num) '_' '3d_img_' num2str(frame) '.png']);
    imwrite(img2, [path '/' num2str(exp_num) '_' '2d_img_' num2str(frame) '.png']);
end
function path = make_dir()
    father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/';
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path = [father_folder DateString];
    mkdir(path);
end
function delta_theta = get_delta_value(visible_pt_3d, params, extrinsic_param, intrinsic_param, activation_label, depth_map)
    k1 = visible_pt_3d(:, 1); k2 = visible_pt_3d(:, 2); plane_ind_set = visible_pt_3d(:, 3); pts_num = size(visible_pt_3d, 1);
    sum_diff = zeros(1, sum(activation_label)); sum_hessian = zeros(sum(activation_label));
    for i = 1 : pts_num
        k = [k1(i) k2(i)]; plane_ind = plane_ind_set(i);
        [grad, diff] = get_grad_value_and_diff(k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label, depth_map);
        sum_diff = sum_diff + diff * grad'; sum_hessian = sum_hessian + grad * grad';
        % check_grad_x(grad_x_params, params, k, plane_ind, activation_label);
        % check_grad_pixel_x(grad_pixel, pts3, extrinsic_param, intrinsic_param);
    end
    delta_theta = smooth_hessian(sum_diff, sum_hessian, activation_label);
end
function delta_theta = smooth_hessian(sum_diff, sum_hessian, activation_label)
    warning('');
    delta_theta = sum_diff * inv(sum_hessian);
    if length(lastwarn) ~= 0
        delta_theta = sum_diff * inv(sum_hessian + eye(sum(activation_label)) * 0.1);
    end
end
function params = update_param(params, delta_theta, activation_label)
    gamma = 0.1;
    params(activation_label) = params(activation_label) + gamma * delta_theta;
end
function diff = calculate_diff(depth_map, intrinsic_param, extrinsic_param, visible_pt_3d, params)
    M = intrinsic_param * extrinsic_param; pts3 = zeros(size(visible_pt_3d, 1), 4);
    for i = 1 : size(pts3, 1)
        pts3(i, :) = [(pts_3d(params, [visible_pt_3d(i, 1) visible_pt_3d(i, 2)], visible_pt_3d(i, 3)))' 1];
    end
    pts2 = project_point_2d(extrinsic_param, intrinsic_param, pts3); depth = (M(3, :) * pts3')';
    gt_depth = zeros(size(pts2, 1), 1);
    for i = 1 : length(gt_depth)
        gt_depth(i) = interpImg(depth_map, pts2(i,:));
    end
    diff = sum((gt_depth - depth).^2); depth_map = add_depth_to_depth_map(depth_map, pts2, 30);
    % show_depth_map(depth_map)
end
function depth_map = add_depth_to_depth_map(depth_map, locations, depth_val)
    locations = round(locations); val_count = 0;
    for i = 1 : size(locations, 1)
        try
            depth_map(locations(i,2), locations(i,1)) = depth_val;
            val_count = val_count + 1;
        catch
        end
    end
end
function show_depth_map(depth_map)
    figure(1); clf; imshow(depth_map / max(max(depth_map)));
end
function [A, diff] = get_grad_value_and_diff(k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label, depth_map)
    M = intrinsic_param * extrinsic_param;
    pts3 = pts_3d(params, k, plane_ind)'; pts2 = project_point_2d(extrinsic_param, intrinsic_param, pts3); depth = M(3, :) * [pts3 1]';
    grad_x_params = get_3d_pt_gradient(params, k, plane_ind, activation_label);
    grad_img = image_grad(depth_map, pts2); grad_pixel = pixel_grad_x(M, pts3); grad_depth = grad_dep(M, pts3);
    A = (grad_depth * grad_x_params - grad_img * grad_pixel * grad_x_params)';
    diff = interpImg(depth_map, pts2) - depth;
    % grad = grad_img * grad_pixel * grad_x_params + grad_depth * grad_x_params;
    %{
    is_right1 = check_grad_depth_params(grad_depth, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label);
    is_right2 = check_grad_pixel_params(grad_pixel, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label);
    if ~(is_right1 && is_right2)
        disp('Error')
    end
    %}
end
function is_right = check_grad_depth_params(grad_depth_x, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label)
    delta = 0.000001; is_right = true; M = intrinsic_param * extrinsic_param; m3 = M(3, :)';
    grad = grad_depth_x * grad_x_params;
    for i = 1 : sum(activation_label)
        if activation_label(i)
            params1 = params; params1(i) = params(i) + delta;
            params2 = params; params2(i) = params(i) - delta;
            pts3_1 = pts_3d(params1, k, plane_ind);
            pts3_2 = pts_3d(params2, k, plane_ind);
            depth1 = m3' * [pts3_1; 1]; depth2 = m3' * [pts3_2; 1];
            re = ((depth1 - depth2) / 2 / delta + delta) ./ (grad(:,i) + delta);
            if max(abs(re - [1 1 1]')) > 0.1
                is_right = false;
            end
        end
    end
end
function is_right = check_grad_pixel_params(grad_pixel, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label)
    delta = 0.000001; is_right = true; M = intrinsic_param * extrinsic_param; m3 = M(3, :)';
    grad = grad_pixel * grad_x_params;
    for i = 1 : sum(activation_label)
        if activation_label(i)
            params1 = params; params1(i) = params(i) + delta;
            params2 = params; params2(i) = params(i) - delta;
            pts3_1 = pts_3d(params1, k, plane_ind);
            pts3_2 = pts_3d(params2, k, plane_ind);
            pixel_loc1 = project_point_2d(extrinsic_param, intrinsic_param, [pts3_1; 1]')';
            pixel_loc2 = project_point_2d(extrinsic_param, intrinsic_param, [pts3_2; 1]')';
            re = ((pixel_loc1 - pixel_loc2) / 2 / delta + delta) ./ (grad(:,i) + delta);
            if max(abs(re - [1 1]')) > 0.1
                is_right = false;
            end
        end
    end
end
function is_right = check_grad_pixel_x(grad_pixel, pts3, extrinsic_param, intrinsic_param)
    delta = 0.0000001; is_right = true; % M = intrinsic_param * extrinsic_param; m1 = M(1, :)'; m2  = M(2, :)'; m3 = M(3, :)';
    for i = 1 : length(pts3)
        pts3_1 = pts3; pts3_1(i) = pts3(i) + delta;
        pts3_2 = pts3; pts3_2(i) = pts3(i) - delta;
        pixel_loc1 = project_point_2d(extrinsic_param, intrinsic_param, [pts3_1 1]);
        pixel_loc2 = project_point_2d(extrinsic_param, intrinsic_param, [pts3_2 1]);
        % pixel_loc1 = [(m1' * [pts3_1 1]') / (m3' * [pts3_1 1]'), (m2' * [pts3_1 1]') / (m3' * [pts3_1 1]')]';
        % pixel_loc2 = [(m1' * [pts3_2 1]') / (m3' * [pts3_2 1]'), (m2' * [pts3_2 1]') / (m3' * [pts3_2 1]')]';
        re = ((pixel_loc1 - pixel_loc2)' / 2 / delta) ./ (grad_pixel(:,i));
        if max(abs(re - [1 1]')) > delta * 10000
            is_right = false;
        end
    end
end
function is_right = check_grad_x(grad_x_params, params, k, plane_ind, activation_label)
    delta = 0.000001; is_right = true;
     for i = 1 : length(activation_label)
         if activation_label(i)
             params1 = params; params1(i) = params(i) + delta;
             params2 = params; params2(i) = params(i) - delta;
             pts3_1 = pts_3d(params1, k, plane_ind);
             pts3_2 = pts_3d(params2, k, plane_ind);
             re = ((pts3_1 - pts3_2) / 2 / delta + delta) ./ (grad_x_params(:,i) + delta);
             if max(abs(re - [1 1 1]')) > delta
                 is_right = false;
             end
         end
     end
end
function params = generate_cubic_params(cuboid)
    theta = cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    params = [theta, xc, yc, l, w, h];
end
function grad = grad_dep(M, x)
    m3 = M(3, 1:3)';
    grad = m3';
end
function grad = pixel_grad_x(M, x)
    m1 = M(1, :)'; m2  = M(2, :)'; m3 = M(3, :)';
    if size(x,1) == 1
        x = x';
    end
    if length(x) < 4
        x = [x; 1];
    end
    gx = m1' / (m3' * x) - m3' * (m1' * x) / (m3' * x)^2; gx = gx(1:3);
    gy = m2' / (m3' * x) - m3' * (m2' * x) / (m3' * x)^2; gy = gy(1:3);
    grad = [gx; gy];
end
function grad = image_grad(image, location)
    x_grad = interpImg(image, [location(1) + 1, location(2)]) - interpImg(image, [location(1), location(2)]);
    y_grad = interpImg(image, [location(1), location(2) + 1]) - interpImg(image, [location(1), location(2)]);
    grad = [x_grad y_grad];
end
function grad = get_3d_pt_gradient(params, k, plane_ind, activation_label)
    grad = zeros(3, 6);
    for i = 1 : length(k)
        if activation_label(1)
            grad(:, 1) = g_theta(params, k, plane_ind);
        end
        if activation_label(2)
            grad(:, 2) = g_xc(params, k, plane_ind);
        end
        if activation_label(3)
            grad(:, 3) = g_yc(params, k, plane_ind);
        end
        if activation_label(4)
            grad(:, 4) = g_l(params, k, plane_ind);
        end
        if activation_label(5)
            grad(:, 5) = g_w(params, k, plane_ind);
        end
        if activation_label(6)
            grad(:, 6) = g_h(params, k, plane_ind);
        end
    end
    grad = grad(:, activation_label);
end
function pts3 = pts_3d(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(:,1)'; k2 = k(:,2)';
    pts3 = zeros(3, 1);
    if plane_ind == 1
        pts3 = [
            xc - 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k1 * cos(theta) * l;
            yc - 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k1 * sin(theta) * l;
            k2 * h
            ];
    end
    if plane_ind == 2
        pts3 = [
            xc + 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k1 * sin(theta);
            yc + 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k1 * cos(theta);
            k2 * h
            ];
    end
    if plane_ind == 3
        pts3 = [
            xc + 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k1 * l * cos(theta);
            yc + 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k1 * l * sin(theta);
            k2 * h
            ];
    end
    if plane_ind == 4
        pts3 = [
            xc - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k1 * sin(theta);
            yc - 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k1 * cos(theta);
            k2 * h
            ];
    end
end
function gtheta = g_theta(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gtheta = zeros(3, 1);
    if plane_ind == 1
        gtheta = [
            1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k1 * l * sin(theta);
            -1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k1 * l * cos(theta);
            0
            ];
    end
    if plane_ind == 2
        gtheta = [
            -1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k1 * cos(theta);
            1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k1 * sin(theta);
            0
            ];
    end
    if plane_ind == 3
        gtheta = [
            -1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k1 * l * sin(theta);
            1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k1 * l * cos(theta);
            0
            ];
    end
    if plane_ind == 4
        gtheta = [
            1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k1 * cos(theta);
            - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k1 * sin(theta);
            0
            ];
    end
end
function gxc = g_xc(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gxc = zeros(3, 1);
    if plane_ind == 1
        gxc = [
            1;
            0;
            0
            ];
    end
    if plane_ind == 2
        gxc = [
            1;
            0;
            0
            ];
    end
    if plane_ind == 3
        gxc = [
            1;
            0;
            0
            ];
    end
    if plane_ind == 4
        gxc = [
            1;
            0;
            0
            ];
    end
end
function gyc = g_yc(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gyc = zeros(3, 1);
    if plane_ind == 1
        gyc = [
            0;
            1;
            0];
    end
    if plane_ind == 2
        gyc = [
            0;
            1;
            0
            ];
    end
    if plane_ind == 3
        gyc = [
            0;
            1;
            0
            ];
    end
    if plane_ind == 4
        gyc = [
            0;
            1;
            0
            ];
    end
end
function gl = g_l(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gl = zeros(3, 1);
    if plane_ind == 1
        gl = [
            -1 / 2 * cos(theta) + k1 * cos(theta);
            -1 / 2 * sin(theta) + k1 * sin(theta);
            0
            ];
    end
    if plane_ind == 2
        gl = [
            1 / 2 * cos(theta);
            1 / 2 * sin(theta);
            0
            ];
    end
    if plane_ind == 3
        gl = [
            1 / 2 * cos(theta) - k1 * cos(theta);
            1 / 2 * sin(theta) - k1 * sin(theta);
            0
            ];
    end
    if plane_ind == 4
        gl = [
            - 1 / 2 * cos(theta);
            - 1 / 2 * sin(theta);
            0
            ];
    end
end
function gw = g_w(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gw = zeros(3, 1);
    if plane_ind == 1
        gw = [
            1 / 2 * sin(theta);
            - 1 / 2 * cos(theta);
            0
            ];
    end
    if plane_ind == 2
        gw = [
            1 / 2 * sin(theta) - k1 * sin(theta);
            -1 / 2 * cos(theta) + k1 * cos(theta);
            0
            ];
    end
    if plane_ind == 3
        gw = [
            -1 / 2 * sin(theta);
            1 / 2 * cos(theta);
            0
            ];
    end
    if plane_ind == 4
        gw = [
            -1 / 2 * sin(theta) + k1 * sin(theta);
            1 / 2 * cos(theta) - k1 * cos(theta);
            0;
            ];
    end
end
function gh = g_h(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gh = zeros(3, 1);
    if plane_ind == 1
        gh = [
            0;
            0;
            k2
            ];
    end
    if plane_ind == 2
        gh = [
            0;
            0;
            k2
            ];
    end
    if plane_ind == 3
        gh = [
            0;
            0;
            k2
            ];
    end
    if plane_ind == 4
        gh = [
            0;
            0;
            k2
            ];
    end
end