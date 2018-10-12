function diff = calculate_diff_pos_v2(depth_map, intrinsic_param, extrinsic_param, visible_pt_3d, params, ratio)
    M = intrinsic_param * extrinsic_param; pts3 = zeros(size(visible_pt_3d, 1), 4);
    for i = 1 : size(pts3, 1)
        pts3(i, :) = [(pts_3d(params, [visible_pt_3d(i, 1) visible_pt_3d(i, 2)], visible_pt_3d(i, 3)))' 1];
    end
    pts2 = project_point_2d(extrinsic_param, intrinsic_param, pts3); depth = (M(3, :) * pts3')';
    gt_depth = zeros(size(pts2, 1), 1);
    for i = 1 : length(gt_depth)
        gt_depth(i) = interpImg(depth_map, pts2(i,:));
    end
    diff = sum((gt_depth - depth).^2); diff = diff * ratio;
    % figure(1); clf;scatter3(pts_3d_record(:,1),pts_3d_record(:,2),pts_3d_record(:,3),3,'r','fill');hold on;axis equal;
    % draw_cubic_shape_frame(cuboid); hold on; scatter3(pts3(:,1),pts3(:,2),pts3(:,3),3,'g','fill');
    % show_depth_map(depth_map)
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