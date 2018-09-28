run('/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Matlab_code/Synthia_3D_scenen_reconstruction/vlfeat-0.9.21/toolbox/vl_setup')
n = 294; error_record = zeros(n, 1); thold = 1e-5; % reconstructed_3d_pts_record = cell(n, 1); affine_matrix_record = cell(n, 1);
exp_re_path = make_dir(); error_recorder1 = zeros(n-1,1); error_recorder2 = zeros(n-1,1); param_record = zeros(n-1, 16);
for frame = 8 : n - 1
    disp(['frame ' num2str(frame) ' finied\n'])
    % f1 = num2str(frame, '%06d'); f2 = num2str(frame + 1, '%06d');
    
    % Get Camera parameter
    % txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt');
    % vec = load(txtPath);
    % extrinsic_params = reshape(vec, 4, 4);
    
    % Get Depth groundtruth
    % depth1 = getDepth(strcat(base_path, GT_Depth_path, f1, '.png')); [label1, instance1] = getIDs(strcat(base_path, GT_seg_path, f1, '.png')); rgb1 = imread(strcat(base_path, GT_RGB_path, f1, '.png'));
    % depth2 = getDepth(strcat(base_path, GT_Depth_path, f2, '.png')); [label2, instance2] = getIDs(strcat(base_path, GT_seg_path, f2, '.png')); rgb2 = imread(strcat(base_path, GT_RGB_path, f2, '.png'));
    [depth1, label1, instance1, rgb1] = readin_images(frame); [depth2, label2, instance2, rgb2] = readin_images(frame + 1);
    [extrinsic_params1, intrinsic_params1] = readin_params(frame); [extrinsic_params2, intrinsic_params2] = readin_params(frame + 1);
    gray1 = rgb2gray(rgb1); gray2 = rgb2gray(rgb2); get_processed_img(gray1, label1);
    [new_img1, fin_ind1] = get_processed_img(gray1, label1); [new_img2, fin_ind2] = get_processed_img(gray2, label2);
    correspondence = find_correspondence_between_img(new_img1, new_img2, fin_ind1, fin_ind2);
    
    cores_pts1 = get_3d_pts(depth1, extrinsic_params1, intrinsic_params1, correspondence(1,:));
    cores_pts2 = get_3d_pts(depth2, extrinsic_params2, intrinsic_params2, correspondence(2,:));
    
    % [param, error] = RANSAC_estimation(cores_pts1, cores_pts2); error_recorder(frame) = error;
    [param1, ~] = RANSAC_estimation(cores_pts1, cores_pts2);
    [param2, ~] = RANSAC_estimation_2(cores_pts1, cores_pts2); % error_recorder(frame) = error;
    % figure(3); scatter3(cores_pts1(:,1),cores_pts1(:,2),cores_pts1(:,3),3,'r','fill'); hold on; scatter3(cores_pts2(:,1),cores_pts2(:,2),cores_pts2(:,3),3,'b','fill')
    
    all_pts1 = get_3d_pts(depth1, extrinsic_params1, intrinsic_params1, fin_ind1');
    all_pts2 = get_3d_pts(depth2, extrinsic_params2, intrinsic_params2, fin_ind2');
    figure(1); clf; scatter3(all_pts1(:,1),all_pts1(:,2),all_pts1(:,3),3,'r','fill'); hold on; scatter3(all_pts2(:,1),all_pts2(:,2),all_pts2(:,3),3,'g','fill')
    all_pts2_transformed1 = (param1 * all_pts2')'; all_pts2_transformed2 = (param2 * all_pts2')'; 
    % [projected_pts2, valid] = projectPoints(all_pts2_transformed, intrinsic_params1(1:3,1:3),extrinsic_params1,[0,0,0,0,0],size(depth1),false);
    % projected_pts2 = projected_pts2(valid, :); lin_ind = sub2ind(size(depth_map), projected_pts2(:,2), projected_pts2(:,1));
    
    figure(1);clf;scatter3(all_pts1(:,1),all_pts1(:,2),all_pts1(:,3),3,'r','fill'); hold on;
    scatter3(all_pts2_transformed1(:,1),all_pts2_transformed1(:,2),all_pts2_transformed1(:,3),3,'b','fill'); hold on; axis equal; [az,el] = view;
    F = getframe(gcf); [X1, Map] = frame2im(F);
    figure(2);clf;scatter3(all_pts1(:,1),all_pts1(:,2),all_pts1(:,3),3,'r','fill'); hold on;
    scatter3(all_pts2_transformed2(:,1),all_pts2_transformed2(:,2),all_pts2_transformed2(:,3),3,'b','fill'); hold on; axis equal; view([az,el]);
    F = getframe(gcf); [X2, Map] = frame2im(F); X2 = imresize(X2, [size(X1,1) size(X1,2)]);
    X = [X1 X2]; imwrite(X, [exp_re_path '/3d_pts_' num2str(frame) '.png']);
    
    [error1, new_image1] = get_error(extrinsic_params1, intrinsic_params1, all_pts2, rgb2, rgb1, param1, fin_ind2');
    [error2, new_image2] = get_error(extrinsic_params1, intrinsic_params1, all_pts2, rgb2, rgb1, param2, fin_ind2');
    imwrite([new_image1 new_image2], [exp_re_path '/rgb_' num2str(frame) '.png']);
    error_recorder1(frame) = error1; error_recorder2(frame) = error2;
    if error1 < error2
        param_record(frame,:) = param1(:);
    else
        param_record(frame,:) = param2(:);
    end
    % scatter3(all_pts2(:,1),all_pts2(:,2),all_pts2(:,3),3,'b','fill');
    % subplot(1,2,1); imshow(rgb1); subplot(1,2,2); imshow(rgb2);
    % Get segmentation mark groudtruth (Instance id looks broken)
    
    
    
    % [road_ix, road_iy] = find(label == 3);
    % linear_ind = sub2ind(size(label), road_ix, road_iy);
    
    % [reconstructed_3d, projects_pts] = get_3d_pts(depth, extrinsic_params, intrinsic_params, linear_ind);
    % origin_pt = (extrinsic_params * [0 0 0 1]')';
    % origin_pt = -(inv(extrinsic_params(1:3,1:3)) * extrinsic_params(1:3,4)); origin_pt = [origin_pt; 1]';
    
    % color = rand([1 3]); origin_pt = (extrinsic_params * [0;0;0;1])';
    % figure(1)
    % scatter3(reconstructed_3d(:,1),reconstructed_3d(:,2),reconstructed_3d(:,3),3,'b','fill')
    % hold on;
    % scatter3(origin_pt(:,1),origin_pt(:,2),origin_pt(:,3),10,'r','fill')
    % scatter3(reconstructed_3d(:,1),reconstructed_3d(:,2),reconstructed_3d(:,3),3,color,'fill')
    % hold on
    % axis equal
    % figure(2)
    % scatter3(origin_pt(:,1),origin_pt(:,2),origin_pt(:,3),3,'r','fill')
    % hold on
    % [affine_matrx, mean_error, param] = estimate_origin_ground_plane(reconstructed_3d);
    % rdx = linspace(min(reconstructed_3d(:,1)), max(reconstructed_3d(:,1)), 100);
    % rdy = linspace(min(reconstructed_3d(:,2)), max(reconstructed_3d(:,2)), 100);
    % [rdX, rdY] = meshgrid(rdx, rdy); rdX = rdX(:); rdY = rdY(:);
    % rdZ = -(param(4) + param(1) * rdX + param(2) * rdY) / param(3);
    % hold on; scatter3(rdX,rdY,rdZ,3,'r','fill')
    
    % diff_pt = origin_pt(1:3) - [rdX(1) rdY(1) rdZ(1)]; diff_pt = [diff_pt 1];
    % d = abs(sum(param .* origin_pt)) / sqrt(sum(param(1:3) .* param(1:3)))
    % reconstructed_3d_pts_record{frame} = reconstructed_3d; affine_matrix_record{frame} = affine_matrx;
    % error_record(frame) = mean_error;
end
write_error(exp_re_path, error_recorder1, error_recorder2);
figure(1); clf; stem(1:length(error_recorder1), [error_recorder1 error_recorder2],'fill','MarkerSize',2); legend('RANSC', 'RANSC + quadratic optimization')
F = getframe(gcf); [X, Map] = frame2im(F); imwrite(X, [exp_re_path '/error_stem' '.png']);
save('adjust_matrix.mat', 'param_record');
function [error, new_img] = get_error(extrinsic, intrinsic, pts3d, rgb_img_from, rgb_img_to, affine_matrix, lin_ind2)
    pts3d_transed = (affine_matrix * pts3d')'; image_size = [size(rgb_img_from, 1) size(rgb_img_from,2)];
    [projected_pts, valid] = projectPoints(pts3d_transed, intrinsic(1:3,1:3),extrinsic,[0,0,0,0,0],[image_size(1)-1 image_size(2)-1],false);
    projected_pts = projected_pts(valid, :); lin_ind1 = sub2ind(image_size, ceil(projected_pts(:,2)), ceil(projected_pts(:,1)));
    lin_ind2 = lin_ind2(valid)';
    % [projected_pts, valid] = projectPoints(pts3d, intrinsic(1:3,1:3),extrinsic,[0,0,0,0,0],image_size,false);
    % projected_pts = projected_pts(valid, :); lin_ind1 = sub2ind(image_size, projected_pts(:,2), projected_pts(:,1));
    
    
    r1 = rgb_img_to(:,:,1); g1 = rgb_img_to(:,:,2); b1 = rgb_img_to(:,:,3);
    r2 = rgb_img_from(:,:,1); g2 = rgb_img_from(:,:,2); b2 = rgb_img_from(:,:,3);
    
    error = sum(abs(r1(lin_ind1) - r2(lin_ind2))) + sum(abs(g1(lin_ind1) - g2(lin_ind2))) + sum(abs(b1(lin_ind1) - b2(lin_ind2)));
    error = error / length(lin_ind1);
    r1(lin_ind1) = r2(lin_ind2); g1(lin_ind1) = g2(lin_ind2); b1(lin_ind1) = b2(lin_ind2);
    % a = zeros(image_size); b = zeros(image_size); c = zeros(image_size);
    % test_img = uint8(zeros([image_size, 3])); a(lin_ind1) = r1(lin_ind1); b(lin_ind1) = g1(lin_ind1); c(lin_ind1) = b1(lin_ind1); 
    % test_img(:,:,1) = a; test_img(:,:,2) = b; test_img(:,:,3) = c; 
    % figure(2); clf; imshow(test_img);
    new_img = uint8(zeros([image_size, 3])); new_img(:,:,1) = r1; new_img(:,:,2) = g1; new_img(:,:,3) = b1; 
    % figure(2); clf; imshow(new_img);
end
function write_error(path, error_recorder1, error_recorder2)
    fileID = fopen([path '/' 'exp_re_recorder.txt'],'w');
    for i = 1 : length(error_recorder1)
        fprintf(fileID, '%5d\t%5d\n', error_recorder1(i), error_recorder2(i));
    end
    fprintf(fileID, 'Average error for RANSC: %5d\nAverage error for RANSC + quadratic optimization: %5d\n', sum(error_recorder1) / length(error_recorder1), sum(error_recorder2) / length(error_recorder2));
    fprintf(fileID, '%% Metric is rgb pixel value differences between buildings and poles in the consecutive frames.\n');
    fclose(fileID);
end
function path = make_dir()
    father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/';
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_');
    path = [father_folder DateString];
    mkdir(path);
end
function [param, error] = RANSAC_estimation(cores_pts1, cores_pts2)
    % Synthia dataset, error comes from Feature points selection procedure
    % Suppose little noise contain within the camera matrix, both extrinsic
    % and intrinsic
    pt_num = 4; tot_it_num = 1000; dist_record = zeros(tot_it_num, 1); frac = 0.5; param_record = zeros(tot_it_num, 16);
    for i = 1 : tot_it_num
        ind = randperm(size(cores_pts1,1), pt_num);
        aff  = cores_pts1(ind, :)' * inv(cores_pts2(ind, :)');
        cur_dist = (cores_pts1 - (aff * cores_pts2')'); cur_dist = cur_dist(:, 1:3);
        cur_dist = sqrt(sum(cur_dist.^2, 2)); sorted_dist = sort(cur_dist); dist_record(i) = sum(sorted_dist(1:ceil(length(sorted_dist)*frac))); param_record(i,:) = aff(:);
    end
    error = min(dist_record) / ceil(size(cores_pts1,1) * frac);
    ind = find(dist_record == min(dist_record)); ind = ind(1); param = reshape(param_record(ind,:), [4,4]);
    % tred_pts = (param * cores_pts2')';
    % figure(1); clf;
    % scatter3(cores_pts1(:,1),cores_pts1(:,2),cores_pts1(:,3),3,'r','fill'); hold on; scatter3(cores_pts2(:,1),cores_pts2(:,2),cores_pts2(:,3),3,'b','fill');
    % scatter3(tred_pts(:,1),tred_pts(:,2),tred_pts(:,3),3,'g','fill'); hold on; 
end
function [param, error] = RANSAC_estimation_2(cores_pts1, cores_pts2)
    % Synthia dataset, error comes from Feature points selection procedure
    % Suppose little noise contain within the camera matrix, both extrinsic
    % and intrinsic
    pt_num = 4; tot_it_num = 1000; dist_record = zeros(tot_it_num, 1); frac = 0.5; param_record = zeros(tot_it_num, 16); frac2 = 0.2;
    for i = 1 : tot_it_num
        ind = randperm(size(cores_pts1,1), pt_num);
        aff  = cores_pts1(ind, :)' * inv(cores_pts2(ind, :)');
        cur_dist = (cores_pts1 - (aff * cores_pts2')'); cur_dist = cur_dist(:, 1:3);
        cur_dist = sqrt(sum(cur_dist.^2, 2)); sorted_dist = sort(cur_dist); dist_record(i) = sum(sorted_dist(1:ceil(length(sorted_dist)*frac))); param_record(i,:) = aff(:);
    end
    error = min(dist_record) / ceil(size(cores_pts1,1) * frac);
    ind = find(dist_record == min(dist_record)); ind = ind(1); param = reshape(param_record(ind,:), [4,4]);
    
    % proj_pts = (param * cores_pts2')'; num_to_estimate = ceil(size(cores_pts1, 1) * frac2);
    % dist = sum((cores_pts1(:, 1:3) - proj_pts(:, 1:3)).^2,2); [~, ind] = sort(dist); to_estimate_ind = ind(1:num_to_estimate);
    % pts1_to_estimate = cores_pts1(to_estimate_ind, :)'; pts2_to_estimate = cores_pts2(to_estimate_ind, :)';
    % a = sum(sum((pts1_to_estimate - (param * pts2_to_estimate)).^2)) / ceil(size(cores_pts1,1) * frac);
    
    
    proj_pts = (param * cores_pts2')'; num_to_estimate = ceil(size(cores_pts1, 1) * frac2);
    dist = sum((cores_pts1(:, 1:3) - proj_pts(:, 1:3)).^2,2); [~, ind] = sort(dist); to_estimate_ind = ind(1:num_to_estimate);
    pts1_to_estimate = cores_pts1(to_estimate_ind, :)'; pts1_to_estimate = pts1_to_estimate(1:3, :);
    pts2_to_estimate = cores_pts2(to_estimate_ind, :)'; pts2_to_estimate = pts2_to_estimate(1:3, :);
    [R_, T_] = ana_sol_for_affine(pts1_to_estimate, pts2_to_estimate); param = zeros(4,4); param(1:3,1:3) = R_; param(1:3,4) = T_; param(4,4) = 1;
    % cur_dist = (cores_pts1 - (aff * cores_pts2')'); cur_dist = cur_dist(:, 1:3);
    % cur_dist = sqrt(sum(cur_dist.^2, 2)); sorted_dist = sort(cur_dist); sum(sorted_dist(1:ceil(length(sorted_dist)*frac))) / ceil(size(cores_pts1,1) * frac);
    % pts1_to_estimate = cores_pts1(to_estimate_ind, :)'; 
    % pts2_to_estimate = cores_pts2(to_estimate_ind, :)';
    % b = sum(sum((pts1_to_estimate - (param * pts2_to_estimate)).^2)) / ceil(size(cores_pts1,1) * frac);
    % a - b
    % b = sum(sum(([pts1_to_estimate; ones(1,78)] - (param * [pts2_to_estimate; ones(1,78)])).^2));
    % tred_pts = (param * cores_pts2')';
    % figure(1); clf; scatter3(cores_pts1(:,1),cores_pts1(:,2),cores_pts1(:,3),3,'r','fill'); hold on;
    % scatter3(proj_pts(:,1),proj_pts(:,2),proj_pts(:,3),3,'g','fill'); hold on; % scatter3(cores_pts2(:,1),cores_pts2(:,2),cores_pts2(:,3),3,'b','fill');
end
function [R_, T_] = ana_sol_for_affine(p, q)
    n = size(p,1); m = size(p,2);
    q = [q;ones(1,size(q,2))]; Q_ = zeros(n+1, n+1);
    for i = 1 : m
        q_ =  q(:, i);
        Q_ = Q_ + q_ * q_';
    end
    c_ = p * q'; A = inv(Q_) * c_'; A = A'; R_ = A(1:3,1:3); T_ = A(:,4);
end
function [extrinsic_params, intrinsic_params] = readin_params(frame)
    base_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    GT_RGB_path = 'RGB/Stereo_Left/Omni_F/';
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath);
    extrinsic_params = reshape(vec, 4, 4);
    focal = 532.7403520000000; cx = 640; cy = 380; % baseline = 0.8;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
end
function [depth, label, instance, rgb] = readin_images(frame)
    base_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    GT_RGB_path = 'RGB/Stereo_Left/Omni_F/';
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    f = num2str(frame, '%06d');
    depth = getDepth(strcat(base_path, GT_Depth_path, f, '.png')); 
    [label, instance] = getIDs(strcat(base_path, GT_seg_path, f, '.png')); 
    rgb = imread(strcat(base_path, GT_RGB_path, f, '.png'));
end
function [new_img, fin_ind] = get_processed_img(img, label)
    if length(size(img)) == 3
        img = rgb2gray(img);
    end
    new_img = uint8(zeros(size(img)));
    building_ind = 2; pole_ind = 7; bool_img = false(size(img)); bool_img([find(label == building_ind); find(label == pole_ind)]) = true;
    SE = strel('square',20); bool_img = imdilate(bool_img, SE); val_ind = find(bool_img == true); new_img(val_ind) = img(val_ind);
    fin_ind = [find(label == building_ind); find(label == pole_ind)];
end
function error = estimate_error(params, pts)
    pts_new = (params * pts')';
    height = pts_new(:, 3) - mean(pts_new(:, 3));
    error = sum(abs(height)) / size(pts, 1);
end
function [affine_matrx, mean_error, param] = estimate_origin_ground_plane(pts)
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

function [reconstructed_3d, projects_pts] = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
    valuable_ind = valuable_ind';
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
    % pt = randn([1 2]);
    pt = [pt, - (param(1) * pt(1) + param(2) * pt(2) + param(4)) / param(3)];
    
end
function correspondence = find_correspondence_between_img(img1, img2, range_ind1, range_ind2)
    [fa,da] = vl_sift(single(img1)); ind1 = sub2ind(size(img1), round(fa(2,:)), round(fa(1,:)));
    [fb,db] = vl_sift(single(img2)); ind2 = sub2ind(size(img2), round(fb(2,:)), round(fb(1,:)));
    [matches, ~] = vl_ubcmatch(da, db); 
    new_ind1 = ind1(matches(1,:)); new_ind2 = ind2(matches(2,:));
    [la1] = ismember(new_ind1, range_ind1); [la2] = ismember(new_ind2, range_ind2);
    la = la1 & la2; new_ind1 = new_ind1(la); new_ind2 = new_ind2(la); correspondence = [new_ind1; new_ind2];
    % img1(new_ind1) = 255; img2(new_ind2) = 255;
    % figure(1);clf;imshow(uint8(img1)); figure(2);clf;imshow(uint8(img2));
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