function [sum_diff, sum_hessian] = accum_diff_and_hessian_pos_v2(visible_pt_3d, params, extrinsic_param, intrinsic_param, activation_label, depth_map, ratio)
    k1 = visible_pt_3d(:, 1); k2 = visible_pt_3d(:, 2); plane_ind_set = visible_pt_3d(:, 3); pts_num = size(visible_pt_3d, 1);
    sum_diff = zeros(1, sum(activation_label)); sum_hessian = zeros(sum(activation_label)); % norm_grad_record = zeros(pts_num, 1);
    for i = 1 : pts_num
        k = [k1(i) k2(i)]; plane_ind = plane_ind_set(i);
        [grad, diff] = get_grad_value_and_diff(k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label, depth_map);
        sum_diff = sum_diff + diff * grad'; sum_hessian = sum_hessian + grad * grad';
        % norm_grad_record(i) = norm(grad);
        % check_grad_x(grad_x_params, params, k, plane_ind, activation_label);
        % check_grad_pixel_x(grad_pixel, pts3, extrinsic_param, intrinsic_param);
    end
    sum_diff = sum_diff * ratio; sum_hessian = sum_hessian * ratio;
    % delta_theta = smooth_hessian(sum_diff, sum_hessian, activation_label);
end
function [A, diff] = get_grad_value_and_diff(k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label, depth_map)
    M = intrinsic_param * extrinsic_param;
    pts3 = pts_3d(params, k, plane_ind)'; pts2 = project_point_2d(extrinsic_param, intrinsic_param, pts3); depth = M(3, :) * [pts3 1]';
    grad_x_params = get_3d_pt_gradient(params, k, plane_ind, activation_label);
    grad_img = image_grad(depth_map, pts2); grad_pixel = pixel_grad_x(M, pts3); grad_depth = grad_dep(M, pts3);
    A = (grad_depth * grad_x_params - grad_img * grad_pixel * grad_x_params)';
    diff = interpImg(depth_map, pts2) - depth;
    % grad = grad_img * grad_pixel * grad_x_params + grad_depth * grad_x_params;
    
    % is_right1 = check_grad_depth_params(grad_depth, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label);
    % is_right2 = check_grad_pixel_params(grad_pixel, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label);
    % if ~(is_right1 && is_right2)
    %    disp('Error')
    % end
    
end
function is_right = check_grad_depth_params(grad_depth_x, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label)
    delta = 0.001; is_right = true; M = intrinsic_param * extrinsic_param; m3 = M(3, :)';
    grad = grad_depth_x * grad_x_params;
    for i = 1 : sum(activation_label)
        if activation_label(i)
            params1 = params; params1(i) = params(i) + delta;
            params2 = params; params2(i) = params(i) - delta;
            pts3_1 = pts_3d(params1, k, plane_ind);
            pts3_2 = pts_3d(params2, k, plane_ind);
            depth1 = m3' * [pts3_1; 1]; depth2 = m3' * [pts3_2; 1];
            re = ((depth1 - depth2) / 2 / delta) ./ (grad(:,i));
            if max(abs(re - [1 1 1]')) > 0.1
                is_right = false;
            end
        end
    end
end
function is_right = check_grad_pixel_params(grad_pixel, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label)
    delta = 0.0000000001; is_right = true; M = intrinsic_param * extrinsic_param; m3 = M(3, :)';
    grad = grad_pixel * grad_x_params;
    for i = 1 : sum(activation_label)
        if activation_label(i)
            params1 = params; params1(i) = params(i) + delta;
            params2 = params; params2(i) = params(i) - delta;
            pts3_1 = pts_3d(params1, k, plane_ind);
            pts3_2 = pts_3d(params2, k, plane_ind);
            pixel_loc1 = project_point_2d(extrinsic_param, intrinsic_param, [pts3_1; 1]')';
            pixel_loc2 = project_point_2d(extrinsic_param, intrinsic_param, [pts3_2; 1]')';
            re = ((pixel_loc1 - pixel_loc2) / 2 / delta) ./ (grad(:,i));
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
        if max(abs(re - [1 1]')) > 0.1
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
             re = ((pts3_1 - pts3_2) / 2 / delta) ./ (grad_x_params(:,i));
             if max(abs(re - [1 1 1]')) > 0.1
                 is_right = false;
             end
         end
     end
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
