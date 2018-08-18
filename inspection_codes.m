% Edited by Sz on 08/18/18
function reconstructed_3d = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
    height = size(depth_map, 1);
    width = size(depth_map, 2);
    x = 1 : height; y = 1 : width;
    [X, Y] = meshgrid(y, x);
    pts = [Y(:) X(:)];
    
    projects_pts = [pts(valuable_ind,2) .* depth_map(valuable_ind), pts(valuable_ind,1) .* depth_map(valuable_ind), depth_map(valuable_ind), ones(length(valuable_ind), 1)];

    reconstructed_3d = (inv(intrinsic_params * extrinsic_params) * projects_pts')';
    % projects_pts = [pts(:,2) .* linear_depth_map(:), pts(:,1) .* linear_depth_map(:), linear_depth_map(:), ones(length(pts(:,1)), 1)];
    % projected_pts_2 = zeros(size(projects_pts));
    %{
    pt_count = 0;
    for i = 1 : height
        for j = 1 : width
            pt_count = pt_count + 1;
            cur_depth = depth_map(i, j);
            pixel_2d = [j * cur_depth, i * cur_depth, cur_depth, 1];
            projected_pts_2(pt_count, :) = (inv(intrinsic_params * extrinsic_params)  * pixel_2d')';
        end
    end
    %}
    %{
    depth_map_reprojected = zeros(height, width);
    real_xy = (intrinsic_params * extrinsic_params * reconstructed_3d')';
    real_xy(:,1) = real_xy(:,1) ./ real_xy(:,3);
    real_xy(:,2) = real_xy(:,2) ./ real_xy(:,3);
    linear_ind = sub2ind(size(depth_map), round(real_xy(:,2)), round(real_xy(:,1)));
    depth_map_reprojected(linear_ind) = real_xy(:, 3);
    max(max(abs(depth_map_reprojected - depth_map)))
    %}
    %{
    for i = 1 : length(linear_depth_map)
        if linear_depth_map(i) ~= depth_map(pts(i,1), pts(i,2))
            disp('error')
            break;
        end
    end
    %}
    %{
    f = sym('f', 'real');
    cx = sym('cx', 'real');
    cy = sym('cy', 'real');
    P = [f 0 cx 0; 0 f cy 0; 0 0 1 0; 0 0 0 1];
    %}
end
%{
% In order to inspect whether choose correct image areas or not
% linear_ind1 = sub2ind([size(label) 3], car_ix, car_iy, ones(length(car_ix), 1) * 1);
% linear_ind2 = sub2ind([size(label) 3], car_ix, car_iy, ones(length(car_ix), 1) * 2);
% linear_ind3 = sub2ind([size(label) 3], car_ix, car_iy, ones(length(car_ix), 1) * 3);
% linear_ind = [linear_ind1;linear_ind2;linear_ind3];
%}