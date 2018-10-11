function best_cuboid = analytical_gradient_multiple_frame(cuboid, depth_map_collection, sampled_visible_pts_set, organized_data)
    % cuboid, intrinsic_param, extrinsic_param, depth_map, lin_ind, visible_pt_3d, pts_3d_record
    params = generate_cubic_params(cuboid);
    activation_label = [1 1 1 1 1 0]; activation_label = (activation_label == 1); it_num = 500; diff_record = zeros(it_num, 1);
    params_record = zeros(it_num, 6);
    for i = 1 : it_num
        sum_diff = 0; sum_hessian = zeros(sum(activation_label)); sum_loss = 0;
        for j = 1 : length(organized_data)
            [visible_pt_3d, extrinsic_param, intrinsic_param, depth_map] = distill_params(sampled_visible_pts_set, organized_data, depth_map_collection, j);
            [accum_diff, accum_hessian] = accum_diff_and_hessian_pos(visible_pt_3d, params, extrinsic_param, intrinsic_param, activation_label, depth_map);
            sum_diff = sum_diff + accum_diff; sum_hessian = sum_hessian + accum_hessian;
            sum_loss = sum_loss + calculate_diff_pos(depth_map, intrinsic_param, extrinsic_param, visible_pt_3d, params);
        end
        delta_theta = smooth_hessian(sum_diff, sum_hessian, activation_label);
        if(judge_stop(delta_theta, params, diff_record))
            break
        end
        params = update_param(params, delta_theta, activation_label);
        params_record(i,:) = params; diff_record(i) = sum_loss;
    end
    params_record = params_record(diff_record~=0, :); diff_record = diff_record(diff_record~=0);
    best_cuboid = select_best(params_record, diff_record, organized_data);
    % [depth_map, re_3d, stem_map] = organize_output_re(params_record, diff_record, sampled_visible_pts_set, depth_map_collection);
end
function org_entry = read_in_org_entry(frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_unioned/';
    ind = num2str(frame, '%06d');
    loaded = load([path ind '.mat']); org_entry = loaded.prev_mark;
end
function best_cuboid = select_best(params_record, diff_record, organized_data)
    num_record = zeros(size(params_record, 1), 1);
    for i = 1 : size(params_record, 1)
        num_record(i) = calculate_tot_num(organized_data, params_record(i,:));
    end
    ave_diff = diff_record ./num_record; min_ind = find(ave_diff == min(ave_diff)); best_params = params_record(min_ind, :);
    best_cuboid = generate_center_cuboid_by_params(best_params); pts_new = organized_data{1}.pts_new;
    % figure(1); clf; draw_cubic_shape_frame(best_cuboid); hold on; scatter3(pts_new(:,1),pts_new(:,2),pts_new(:,3),3,'g','fill')
end
function tot_num = calculate_tot_num(organized_data, params)
    tot_num = 0;
    for i = 1 : length(organized_data)
        cuboid = generate_center_cuboid_by_params(params); pts_2d = organized_data{i}.pts_new; pts_2d = [pts_2d(:,1:2) ones(size(pts_2d,1),1)];
        tot_num = tot_num + judge_pts_in_cuboid(cuboid, pts_2d, organized_data{i}.pts_new);
    end
end
function num = judge_pts_in_cuboid(cuboid, pts_sampled, pts_org)
    lose_fac = 0.0001;
    corner_points = cuboid{5}.pts(:,1:2);
    org_pt = [corner_points(1,:) 1]; pt_x = [corner_points(2,:) 1]; pt_y = [corner_points(4,:), 1];
    l = norm(pt_x - org_pt); w = norm(pt_y - org_pt);
    pts_old = [org_pt;pt_x;pt_y;]; pts_new = [0, 0, 1; l, 0, 1; 0, w, 1;]; A = pts_new' * smooth_inv(pts_old');
    pts_sampled_transed = (A * pts_sampled')';
    selector = (pts_sampled_transed(:,1) <= l + lose_fac) & (pts_sampled_transed(:,1) >= 0 - lose_fac) & (pts_sampled_transed(:,2) <= w + lose_fac) & (pts_sampled_transed(:,2) >= 0 - lose_fac);
    num = sum(selector); in_pts = pts_org(selector, :); out_pts = pts_org(~selector, :);
    % figure(1); clf; draw_cubic_shape_frame(cuboid); hold on; scatter3(in_pts(:,1),in_pts(:,2),in_pts(:,3),3,'g','fill');
    % hold on; scatter3(out_pts(:,1),out_pts(:,2),out_pts(:,3),3,'r','fill');
end
function A_inv = smooth_inv(A)
    warning(''); size_A = size(A,1);
    A_inv = inv(A);
    if length(lastwarn) ~= 0
        A_inv = inv(A + eye(size_A) * 0.1);
    end
end
function [visible_pt_3d, extrinsic_param, intrinsic_param, depth_map] = distill_params(sampled_visible_pts_set, organized_data, depth_map_collection, frame)
    visible_pt_3d = sampled_visible_pts_set{frame}; visible_pt_3d = visible_pt_3d(:,4:6); visible_pt_3d = [visible_pt_3d(:,2:3), visible_pt_3d(:,1)];
    extrinsic_param = organized_data{frame}.extrinsic_params * inv(organized_data{frame}.affine_matrx);
    intrinsic_param = organized_data{frame}.intrinsic_params; depth_map = depth_map_collection{frame};
end
function [best_params, depth_map, re_3d, stem_map, max_iou] = organize_output_re(params_record, cuboid_gt, diff_record, visible_pt_3d, depth_map, extrinsic_param, intrinsic_param)
    [best_cuboid, max_iou, best_params] = find_best_cuboid(params_record, cuboid_gt);
    stem_map = plot_stem(diff_record);
    re_3d = plot_scene(cuboid_gt, best_params, visible_pt_3d);
    re_2d = visualize_on_depth_map(depth_map, best_params, visible_pt_3d, extrinsic_param, intrinsic_param);
end
function [best_cuboid, max_iou, best_params] = find_best_cuboid(params_record, cuboid_gt)
    iou_record = zeros(length(params_record), 1);
    for i = 1 : length(params_record)
        cuboid_cur = generate_center_cuboid_by_params(params_record(i,:));
        iou_record(i) = calculate_IOU(cuboid_gt, cuboid_cur);
    end
    ind_max = find(iou_record == max(iou_record)); ind_max = ind_max(1);
    best_cuboid = generate_center_cuboid_by_params(params_record(ind_max,:));
    max_iou = max(iou_record); best_params = params_record(ind_max,:);
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
    params_th = 0.1; is_stop = false; diff_record = diff_record(diff_record~=0);
    step_range = 10; th_hold = 10;
    if sum((params(4:5))) < params_th
        is_stop = true;
    end
    if length(diff_record) > step_range
        if abs(diff_record(end) - diff_record(end - step_range)) < th_hold
            is_stop = true;
        end
    end
end
function X = plot_stem(diff_record)
    f = figure('visible', 'off'); stem(diff_record(diff_record~=0), 'fill'); F = getframe(f); [X, Map] = frame2im(F);
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


function show_depth_map(depth_map)
    figure(1); clf; imshow(depth_map / max(max(depth_map)));
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
function cuboid = generate_center_cuboid_by_params(params)
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
end
