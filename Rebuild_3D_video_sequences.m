env_set()
function env_set()
    base_path = '/home/ray/Downloads/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    % baseline = 0.8;
    focal = 532.7403520000000;
    cx = 640;
    cy = 380;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1];
    intrinsic_params(4,4) = 1;
    n = 80;
    frame = 0;
    gt = struct;
    
    for i = 1 : n
        
        f = num2str(frame, '%06d');
        
        % Get Camera parameter
        txtPath = strcat(base_path, cam_para_path, num2str((i-1), '%06d'), '.txt');
        vec = load(txtPath);
        gt(i).extrinsic_params = reshape(vec, 4, 4)';
        gt(i).intrinsic_params = intrinsic_params;
        
        % Get Depth groundtruth
        ImagePath = strcat(base_path, GT_Depth_path, f, '.png');
        gt(i).depth = getDepth(ImagePath);
        
        % Get segmentation mark groudtruth (Instance id looks broken)
        ImagePath = strcat(base_path, GT_seg_path, f, '.png');
        gt(i).seg = getIDs(ImagePath);
        
        frame = frame + 1;
        
        reconstructed_3d = get_3d_pts(gt(i).depth, gt(i).extrinsic_params, gt(i).intrinsic_params);
        
        figure(1)
        scatter3(reconstructed_3d(:,1),reconstructed_3d(:,2),reconstructed_3d(:,3),3,'r');
    end
end
function reconstructed_3d = get_3d_pts(depth_map, extrinsic_params, intrinsic_params)
    height = size(depth_map, 1);
    width = size(depth_map, 2);
    x = 1 : height; y = 1 : width;
    [X, Y] = meshgrid(x, y);
    pts = [X(:), Y(:)];
    linear_depth_map = depth_map';
    linear_depth_map = linear_depth_map(:);
    
    projects_pts = [pts(:,1) .* linear_depth_map, pts(:,2) .* linear_depth_map, linear_depth_map, ones(length(linear_depth_map), 1)];
    reconstructed_3d = (inv(intrinsic_params * extrinsic_params) * projects_pts')';
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