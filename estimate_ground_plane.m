function [affine_matrx, mean_error] = estimate_ground_plane(frame)
    intrinsic_params = get_intrinsic_matrix();
    [extrinsic_params, depth, label, ~] = grab_provided_data(frame);
    
    road_label = 3; 
    road_pt_3d = acquire_3d_reconstructed_pts(extrinsic_params, intrinsic_params, depth, label, road_label);    
    [affine_matrx, ~, mean_error] = estimate_plane(road_pt_3d);
end

function [affine_matrx, org_error, current_error] = estimate_plane(org_pts)
    [affine_matrx, ~] = quadrtic_plane_optimization(org_pts); 
    
    flat_pts1 = (affine_matrx * org_pts')'; org_error = sum(abs(flat_pts1(:,3))) / size(org_pts, 1);
    robust_pts = get_robust_points(flat_pts1, org_pts);
    
    [affine_matrx, ~] = quadrtic_plane_optimization(robust_pts); 
    flat_pts2 = (affine_matrx * org_pts')'; current_error = sum(abs(flat_pts2(:,3))) / size(org_pts, 1);
end
function reconstructed_3d = acquire_3d_reconstructed_pts(extrinsic_params, intrinsic_params, depth, label, coverred_label)
    selected = false(size(label));
    for i = 1 : length(coverred_label)
        selected = selected | (label == coverred_label(i));
    end
    [ix, iy] = find(selected); 
    linear_ind = sub2ind(size(label), ix, iy); 
    [reconstructed_3d, ~] = get_3d_pts(depth, extrinsic_params, intrinsic_params, linear_ind);
end
function selected_pts = get_robust_points(flat_pts, org_pts)
    height = flat_pts(:,3); rank = 0.95;
    [~, ind] = sort(abs(height)); ind = ind(1 : ceil(length(ind) * rank));
    selected_pts = org_pts(ind, :);
end
function save_fig(frame, type)
    father_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/ground_plane_new/';
    if type == 1
        % Ground Plane
        F = getframe(gcf); [X, ~] = frame2im(F); imwrite(X, [father_path 'ground_plane_' num2str(frame) '.png']);
    elseif type == 2
        % Other points
        F = getframe(gcf); [X, ~] = frame2im(F); imwrite(X, [father_path 'other_points_' num2str(frame) '.png']);
    elseif type == 3
        % Error figure
        savefig([father_path 'erre_figure' '.fig'])
        F = getframe(gcf); [X, ~] = frame2im(F); imwrite(X, [father_path 'error_figure' '.png']);
    end
end
function draw_pts(flat_pts)
    clf; scatter3(flat_pts(:,1),flat_pts(:,2),flat_pts(:,3),3,'r','filled'); axis equal;
end
function [extrinsic_params, depth, label, instance] = grab_provided_data(frame)
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
end
function error = estimate_error(params, pts)
    pts_new = (params * pts')'; 
    height = pts_new(:, 3);
    error = sum(abs(height)) / size(pts, 1);
end
function [affine_matrx, plane_param] = quadrtic_plane_optimization(pts)
    mean_pts = mean(pts);
    sum_mean_xy = sum((pts(:,1) - mean_pts(1)) .* (pts(:,2) - mean_pts(2)));
    sum_mean_x2 = sum((pts(:,1) - mean_pts(1)).^2);
    sum_mean_y2 = sum((pts(:,2) - mean_pts(2)).^2);
    sum_mean_xz = sum((pts(:,1) - mean_pts(1)) .* (pts(:,3) - mean_pts(3)));
    sum_mean_yz = sum((pts(:,2) - mean_pts(2)) .* (pts(:,3) - mean_pts(3)));
    M = [sum_mean_x2 sum_mean_xy; sum_mean_xy sum_mean_y2];
    N = [sum_mean_xz; sum_mean_yz];
    param_intermediate = inv(M) * N;
    A = param_intermediate(1); B = param_intermediate(2);
    plane_param = [A, B, -1, -A*mean_pts(1)-B*mean_pts(2)+mean_pts(3)];
    affine_matrx = get_affine_transformation_from_plane(plane_param, pts);
    % mean_error = sum(abs(plane_param * pts')) / norm(plane_param) / size(pts, 1);
end

function [reconstructed_3d, projects_pts] = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
    height = size(depth_map, 1);
    width = size(depth_map, 2);
    x = 1 : height; y = 1 : width;
    [X, Y] = meshgrid(y, x);
    pts = [Y(:) X(:)];
    projects_pts = [pts(valuable_ind,2) .* depth_map(valuable_ind), pts(valuable_ind,1) .* depth_map(valuable_ind), depth_map(valuable_ind), ones(length(valuable_ind), 1)];
    reconstructed_3d = (inv(intrinsic_params * extrinsic_params) * projects_pts')';
end
function affine_transformation = get_affine_transformation_from_plane(param, pts)
    origin = mean(pts); origin = origin(1:3);
    dir1 = (rand_sample_pt_on_plane(param, true) - rand_sample_pt_on_plane(param, false)); dir1 = dir1 / norm(dir1);
    dir3 = param(1:3); dir3 = dir3 / norm(dir3);
    dir2 = cross(dir1, dir3); dir2 = dir2 / norm(dir2);
    dir =[dir1;dir2;dir3];
    affine_transformation = get_affine_transformation(origin, dir);
end
function pt = rand_sample_pt_on_plane(param, flag)
    if flag
        pt = [0.2946 -3.0689];
    else
        pt = [0.9895 -1.8929];
    end
    pt = [pt, - (param(1) * pt(1) + param(2) * pt(2) + param(4)) / param(3)];
end

function affine_transformation = get_affine_transformation(origin, new_basis)
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
end