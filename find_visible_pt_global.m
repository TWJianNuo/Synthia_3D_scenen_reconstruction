function visible_pt_label = find_visible_pt_global(cubics, pts_3d, intrinsic_params, extrinsic_params)
    if size(pts_3d, 2) < 4
        pts_3d = [pts_3d, ones(size(pts_3d, 1), 1)];
    end
    num_pt = size(pts_3d, 1); cam_origin = inv(extrinsic_params) * [0 0 0 1]';
    M = inv(intrinsic_params * extrinsic_params); pts_2d = (intrinsic_params * extrinsic_params * pts_3d')'; pts_2d(:, 1) = pts_2d(:, 1) ./ pts_2d(:, 3); pts_2d(:, 2) = pts_2d(:, 2) ./ pts_2d(:, 3); pts_2d = pts_2d(:, 1:2);
    visible_pt_label = false(num_pt, 1); deviation_threshhold = 0.00001; valid_plane_num = 4; num_obj = size(cubics, 1);
    for ii = 1 : num_pt
        single_pt_all_possible_pos = zeros(valid_plane_num * num_obj, 4);
        valid_label = false(valid_plane_num * num_obj, 1);
        for k = 1 : num_obj
            cuboid = cubics{k};
            for i = 1 : valid_plane_num
                params = cuboid{i}.params;
                z = - params * M(:, 4) / (pts_2d(ii, 1) * params * M(:, 1) + pts_2d(ii, 2) * params * M(:, 2) + params * M(:, 3));
                single_pt_all_possible_pos((k - 1) * valid_plane_num + i, :) = (M * [pts_2d(ii, 1) * z pts_2d(ii, 2) * z z 1]')';
            end
            [valid_label((k-1) * valid_plane_num + 1 : k * valid_plane_num, :), ~] = judge_on_cuboid(cuboid, single_pt_all_possible_pos((k - 1) * valid_plane_num + 1 : k * valid_plane_num, :)); 
        end
        if length(single_pt_all_possible_pos(valid_label)) > 0
            vale_pts = single_pt_all_possible_pos(valid_label, :);
            dist_to_origin = sum((vale_pts(:, 1:3) - cam_origin(1:3)').^2, 2);
            shortest_ind = find(dist_to_origin == min(dist_to_origin));
            shortest_ind = shortest_ind(1);
            if(sum((vale_pts(shortest_ind, 1:3) - pts_3d(ii, 1:3)).^2) < deviation_threshhold)
                visible_pt_label(ii) = true;
            end
        end
    end
    pts_2d = pts_2d(visible_pt_label, :);
end

function [valid_label, type] = judge_on_cuboid(cuboid, pts)
    valid_label = false([length(pts) 1]);
    type = ones([length(pts) 1]) * (-1);
    th = 0.00001;
    if size(pts, 2) == 3
        pts = [pts ones(length(pts), 1)];
    end
    for i = 1 : 4
        pts_local_coordinate = (cuboid{i}.T * pts')';
        jdg_re = (pts_local_coordinate(:, 1) > -th & pts_local_coordinate(:, 1) < cuboid{i}.length1 + th) & (pts_local_coordinate(:, 3) > 0 - th & pts_local_coordinate(:, 3) < cuboid{i}.length2 + th) & (abs(pts_local_coordinate(:, 2)) < th);
        valid_label = valid_label | jdg_re;
        type(jdg_re) = i;
    end
end