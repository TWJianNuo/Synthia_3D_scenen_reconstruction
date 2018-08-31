function [re, new_affine_matrix]= judge_plane_estimation(frame, affine_matrix)
    % Road  3
    % frame = 2;
    base_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    GT_RGB_path = 'RGB/Stereo_Left/Omni_F/';
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    
    ground_dist_threshold = 1;
    
    focal = 532.7403520000000; cx = 640; cy = 380; % baseline = 0.8;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
    
    n = 294;
    
    f = num2str(frame, '%06d');
    
    re = false;
    % Get Camera parameter
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt');
    vec = load(txtPath);
    extrinsic_params = reshape(vec, 4, 4);
    
    % Get Depth groundtruth
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png');
    depth = getDepth(ImagePath);
    
    % Get segmentation mark groudtruth (Instance id looks broken)
    ImagePath = strcat(base_path, GT_seg_path, f, '.png');
    [label, ~] = getIDs(ImagePath);
    
    
    [road_ix, road_iy] = find(label == 3);
    linear_ind = sub2ind(size(label), road_ix, road_iy);
    
    reconstructed_3d = get_3d_pts(depth, extrinsic_params, intrinsic_params, linear_ind);
    new_pts = (affine_matr * reconstructed_3d')'; sum_dis_to_ground = sum(new_pts(:,3).^2) / size(new_pts, 1);
    if sum_dis_to_ground < groundTruth_dist_threshold
        new_affine_matrix = affine_matrix;
        re = true;
    else
        [new_affine_matrix, mean_error] = estimate_origin_ground_plane(reconstructed_3d);
        re = false;
    end
    % Check:
    % img = imread(strcat(base_path, GT_RGB_path, num2str((frame-1), '%06d'), '.png'));
end
function [affine_matrx, mean_error] = estimate_origin_ground_plane(pts)
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
    param = [A, B, -1, -A*mean_pts(1)-B*mean_pts(2)+mean_pts(3)];
    affine_matrx = get_affine_transformation_from_plane(param, pts);
    mean_error = sum((param * pts').^2) / size(pts, 1);
end
function reconstructed_3d = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
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
    dir1 = (rand_sample_pt_on_plane(param) - rand_sample_pt_on_plane(param)); dir1 = dir1 / norm(dir1);
    dir3 = param(1:3); dir3 = dir3 / norm(dir3);
    dir2 = cross(dir1, dir3); dir2 = dir2 / norm(dir2);
    dir =[dir1;dir2;dir3];
    affine_transformation = get_affine_transformation(origin, dir);
end
function pt = rand_sample_pt_on_plane(param)
    pt = randn([1 2]); pt = [pt, - (param(1) * pt(1) + param(2) * pt(2) + param(4)) / param(3)];
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
    % Check: 
    % (affine_transformation * [old_pts ones(3,1)]')'
end