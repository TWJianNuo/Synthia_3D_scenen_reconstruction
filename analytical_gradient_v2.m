function fin_param = analytical_gradient_v2(cuboid, P, T, visible_pt_3d, depth_map, gt_pt_3d, to_plot)
    % This is an edited version of the gradient algorithm
    theta = cuboid{1}.theta; close all
    l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2;
    center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    M = P * T;
    depth_map = [depth_map depth_map(:, end)]; depth_map = [depth_map; depth_map(end, :)];
    img_height = size(depth_map, 1); img_width = size(depth_map, 2);
    gamma = [0.8 0.8 0.8 0.8 0.8 0.01]; max_it_num = 500; terminate_flag = true; terminate_condition = 1e-4;
    
    activation_label = [1 1 1 1 1 0]; activation_label = (activation_label == 1);
    
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
    gradient_set{1} = gra_pts_3d_theta; gradient_set{2} = gra_pts_3d_xs;    gradient_set{3} = gra_pts_3d_ys;
    gradient_set{4} = gra_pts_3d_l;     gradient_set{5} = gra_pts_3d_w;     gradient_set{6} = gra_pts_3d_h;
    
    k1 = visible_pt_3d(:, 4); k2 = visible_pt_3d(:, 5);
    
    px_ = @(pt_affine_3d)round((M(1, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
    py_ = @(pt_affine_3d)round((M(2, :) *  pt_affine_3d') / (M(3, :) * pt_affine_3d'));
    estimated_depth_ = @(pt_affine_3d) M(3, :) * pt_affine_3d';
    diff_ = @(pt_affine_3d, gt) gt - estimated_depth_(pt_affine_3d);
    
    diff_record = zeros(100, 1); depth_cpy = depth_map;
    
    gt_record = zeros(length(k1), 1);
    for i = 1 : length(k1)
        try
            pt_affine_3d = gt_pt_3d(i, :);
            px = px_(pt_affine_3d); py = py_(pt_affine_3d);
            gt_record(i) = depth_map(py, px);
        catch
            if px <= 0
                px = 1;
            end
            if px >= img_width
                px = img_width;
            end
            if py <= 0
                py = 1;
            end
            if py >= img_height
                py = img_height;
            end
            gt_record(i) = depth_map(py, px);
        end
    end
    delta_record = zeros(70, 1);
    
    for it_num = 1 : max_it_num
        tot_diff_record = 0; first_order = 0; hessian = 0; pt_affine_3d_record = zeros(length(k1), 3);
        for i = 1 : length(k1)
            plane_ind = visible_pt_3d(i, 6);
            % Calculate Diff_val
            pt_affine_3d = [pts_3d{plane_ind}(theta, xc, yc, l, w, h, k1(i), k2(i)); 1]';
            pt_affine_3d_record(i, :) = pt_affine_3d(1:3); % px = px_(pt_affine_3d); py = py_(pt_affine_3d);
            % depth_cpy(py, px) = 40;
            try
                diff = diff_(pt_affine_3d, gt_record(i)); 
                tot_diff_record = tot_diff_record + diff^2;
            catch
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
            J = J_3;
            
            hessian = hessian + J' * J;
            first_order = first_order + diff * J';
        end
        %{
        figure(1)
        clf
        cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
        draw_cubic_shape_frame(cuboid)
        hold on
        scatter3(gt_pt_3d(:,1),gt_pt_3d(:,2),gt_pt_3d(:,3),12,'g','fill')
        hold on
        scatter3(pt_affine_3d_record(:,1),pt_affine_3d_record(:,2),pt_affine_3d_record(:,3),9,'r','fill')
        hold on
        scatter3(to_plot(:,1), to_plot(:,2), to_plot(:,3), 3, 'k', 'fill')
        axis equal; view(-46.3, 23.6)
        %}
        % F = getframe(gcf);
        % [X, Map] = frame2im(F);
        % imwrite(X, ['/home/ray/Desktop/exp_new/' num2str(it_num) '.png'])

        delta = (hessian + eye(size(hessian,1))) \ first_order;
        cur_params = [theta xc yc l w 0]; cur_params(activation_label) = cur_params(activation_label) + delta' .* gamma(activation_label);
        theta = cur_params(1); xc = cur_params(2); yc = cur_params(3); l = cur_params(4); w = cur_params(5);
        diff_record(it_num) = tot_diff_record; delta_record(it_num) = max(abs(delta));
        if max(abs(delta)) < terminate_condition
            break;
        end
    end
    fin_param = [xc, yc, theta, l, w, h]; 
    % figure(2); 
    % stem(delta_record);
    % figure(3)
    % stem(diff_record);
end
function show_depth_map(depth_map)
    imshow(uint16(depth_map * 1000));
end