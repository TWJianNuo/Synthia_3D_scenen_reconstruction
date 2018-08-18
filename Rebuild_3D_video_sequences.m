env_set()
% Label info below
% Class         ID
% Void          0
% Sky           1
% Building      2
% Road          3
% Sidewalk      4
% Fence         5
% Vegetation    6
% Pole          7
% Car           8
% Traffic Sign  9
% Pedestrian    10
% Bicycle       11
% Lanemarking	12	
% Reserved		13
% Reserved      14
% Traffic Light	15
function env_set()
    base_path = '/home/ray/ShengjieZhu/Fall Semester/synthia-dataset/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    GT_RGB_path = 'RGB/Stereo_Left/Omni_F/';
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
    
    focal = 532.7403520000000; cx = 640; cy = 380; % baseline = 0.8;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
    
    n = 958;
    
    for frame = 1 : n
        f = num2str(frame, '%06d');
        
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
        img = imread(strcat(base_path, GT_RGB_path, num2str((frame-1), '%06d'), '.png'));
        
        [car_ix, car_iy] = find(label == 8);
        linear_ind = sub2ind(size(label), car_ix, car_iy);
        
        reconstructed_3d = get_3d_pts(depth, extrinsic_params, intrinsic_params, linear_ind);
        show_img_on_index(img, 1);
        figure(2)
        scatter3(reconstructed_3d(:,1),reconstructed_3d(:,2),reconstructed_3d(:,3),3,'r');
    end
end
function show_img_on_index(image, figure_ind)
    figure(figure_ind)
    imshow(image)
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