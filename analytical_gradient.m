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

