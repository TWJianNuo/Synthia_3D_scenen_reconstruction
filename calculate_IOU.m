function IOU = calculate_IOU(cuboid_gt, cuboid_cur)
    overlap_ratio = judge_overlap_ratio(cuboid_gt, cuboid_cur);
    size_gt = get_size(cuboid_gt); size_cur = get_size(cuboid_cur);
    IOU = overlap_ratio * size_cur / (size_cur + size_gt - overlap_ratio * size_cur);
end
function overlap_ratio = judge_overlap_ratio(cuboid_gt, cuboid_cur)
    pts_sampled = sample_cubic_ground(cuboid_cur); lose_fac = 0.0001;
    corner_points = cuboid_gt{5}.pts(:,1:2);
    org_pt = [corner_points(1,:) 1]; pt_x = [corner_points(2,:) 1]; pt_y = [corner_points(4,:), 1];
    l = norm(pt_x - org_pt); w = norm(pt_y - org_pt);
    pts_old = [org_pt;pt_x;pt_y;]; pts_new = [0, 0, 1; l, 0, 1; 0, w, 1;]; A = pts_new' * smooth_inv(pts_old');
    pts_sampled_transed = (A * pts_sampled')';
    selector = (pts_sampled_transed(:,1) <= l + lose_fac) & (pts_sampled_transed(:,1) >= 0 - lose_fac) & (pts_sampled_transed(:,2) <= w + lose_fac) & (pts_sampled_transed(:,2) >= 0 - lose_fac);
    overlap_ratio = sum(selector) / size(pts_sampled, 1);
    % draw_rectangle((A * [corner_points ones(size(corner_points, 1), 1)]')');
    % figure(1); clf; scatter(pts_sampled_transed(:,1),pts_sampled_transed(:,2),10,'r','fill');
end
function rect_size = get_size(cuboid)
    l = cuboid{1}.length1; w = cuboid{2}.length1;
    rect_size = l * w;
end
function pts_old = sample_cubic_ground(cuboid)
    corner_points = cuboid{5}.pts(:,1:2); edge_sample_num = 10;
    org_pt = [corner_points(1,:) 1]; pt_x = [corner_points(2,:) 1]; pt_y = [corner_points(4,:), 1];
    l = norm(pt_x - org_pt); w = norm(pt_y - org_pt);
    pts_old = [org_pt;pt_x;pt_y;]; pts_new = [0, 0, 1; l, 0, 1; 0, w, 1;];
    A = pts_old' * smooth_inv(pts_new'); [xx, yy] = meshgrid(linspace(0, l, edge_sample_num), linspace(0, w, edge_sample_num));
    pts_new = [xx(:) yy(:) ones(length(xx(:)),1)]; pts_old = (A * pts_new')';
    % figure(1); clf; draw_rectangle(corner_points); scatter(pts_old(:,1),pts_old(:,2),10,'r','fill');
end
function A_inv = smooth_inv(A)
    warning(''); size_A = size(A,1);
    A_inv = inv(A);
    if length(lastwarn) ~= 0
        A_inv = inv(A + eye(size_A) * 0.1);
    end
end
function draw_rectangle(pts)
    for i = 1 : size(pts,1)
        ind1 = i; ind2 = i + 1; 
        if ind2 > size(pts,1) 
            ind2 = 1; 
        end
        pts_set = [pts(ind1, 1:2);pts(ind2, 1:2);];
        plot(pts_set(:,1), pts_set(:,2),'b'); hold on;
    end
end