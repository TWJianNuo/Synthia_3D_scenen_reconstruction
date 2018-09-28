function generate_world_coordinate_system_transformation_function()
    intrinsic_params = get_intrinsic_matrix(); n = 294;
    affine_matrix_record = cell(n, 1); threshold = 0.1;
    for frame = 1 : n
        [extrinsic_params, depth, label, building_instance, rgb] = grab_provided_data(frame);
        reconstructed_3d = acquire_3d_reconstructed_pts(extrinsic_params, intrinsic_params, depth, label, coverred_label);
        if frame >= 2
            transferred_pts_trail = affine_matrix_record{}
        else
            [affine_matrx, ~] = Estimate_ground_plane(frame);
            affine_matrix_record{frame} = affine_matrx;
        end
    end
end

function [extrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame)
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
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

function [reconstructed_3d, projects_pts] = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
    height = size(depth_map, 1);
    width = size(depth_map, 2);
    x = 1 : height; y = 1 : width;
    [X, Y] = meshgrid(y, x);
    pts = [Y(:) X(:)];
    projects_pts = [pts(valuable_ind,2) .* depth_map(valuable_ind), pts(valuable_ind,1) .* depth_map(valuable_ind), depth_map(valuable_ind), ones(length(valuable_ind), 1)];
    reconstructed_3d = (inv(intrinsic_params * extrinsic_params) * projects_pts')';
end