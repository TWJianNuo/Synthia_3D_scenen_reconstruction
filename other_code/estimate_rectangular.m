function [params, cuboid] = estimate_rectangular(pts_3d)
    pts_2d = pts_3d(:, 1:2);
    % pts_2d = [pts_2d ones(size(pts_2d, 1))];
    
    R_ = @(theta) [cos(theta) -sin(theta); sin(theta) cos(theta)];
    theta_stack = 0 : 89;
    dist_record = zeros(length(theta_stack), 1);
    rectangles_record = cell(length(theta_stack), 1);
    for i = 1 : length(theta_stack)
        theta = theta_stack(i);
        cur_R = R_(theta / 180 * pi);
        cur_inv_R = R_(-theta / 180 * pi);
        cur_rotated_pts_2d = (cur_R * pts_2d')';
        rectangular_init_guess = zeros(4, 2);
        rectangular_init_guess(1, :) = [min(cur_rotated_pts_2d(:, 1)), min(cur_rotated_pts_2d(:, 2))];
        rectangular_init_guess(2, :) = [max(cur_rotated_pts_2d(:, 1)), min(cur_rotated_pts_2d(:, 2))];
        rectangular_init_guess(3, :) = [max(cur_rotated_pts_2d(:, 1)), max(cur_rotated_pts_2d(:, 2))];
        rectangular_init_guess(4, :) = [min(cur_rotated_pts_2d(:, 1)), max(cur_rotated_pts_2d(:, 2))];
        cur_l = max(cur_rotated_pts_2d(:, 1)) - min(cur_rotated_pts_2d(:, 1));
        cur_w = max(cur_rotated_pts_2d(:, 2)) - min(cur_rotated_pts_2d(:, 2));
        cur_theta = -theta / 180 * pi;
        rectangular_guess_cornerpts = (cur_inv_R * rectangular_init_guess')';
        rectangle = generate_rectangle(mean(rectangular_guess_cornerpts), cur_l, cur_w, cur_theta);
        sum_dist = calculate_sum_dist(pts_2d, rectangle);
        dist_record(i) = sum_dist;
        rectangles_record{i} = rectangle;
    end
    [val, ind] = min(dist_record);
    rectangle = rectangles_record{ind};
    l = rectangle.scale(1); w = rectangle.scale(2); h = max(pts_3d(:,3));
    theta = rectangle.theta; cx = rectangle.center(1); cy = rectangle.center(2);
    params = [cx, cy, theta, l, w, h];
    cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
end
function rectangle = generate_rectangle(center_pt, l, w, theta)
    % center_pt = randi([1 10], [2 2]);
    x = center_pt(1); y = center_pt(2);
    % w = randi([1 10], 1);
    % l = randi([1 10], 1);
    % theta = rand(1) * pi;
    
    rectangle.center = center_pt;
    rectangle.scale = [l w];
    
    pt1 = [x - l / 2 * cos(theta) + w / 2 * sin(theta); y - l / 2 * sin(theta) - w / 2 * cos(theta)]';
    pt2 = [x + l / 2 * cos(theta) + w / 2 * sin(theta); y + l / 2 * sin(theta) - w / 2 * cos(theta)]';
    pt3 = [x + l / 2 * cos(theta) - w / 2 * sin(theta); y + l / 2 * sin(theta) + w / 2 * cos(theta)]';
    pt4 = [x - l / 2 * cos(theta) - w / 2 * sin(theta); y - l / 2 * sin(theta) + w / 2 * cos(theta)]';
    rectangle.pts = [pt1;pt2;pt3;pt4];
    
    line1 = [-sin(theta), cos(theta), x * sin(theta) - y * cos(theta) + w / 2];
    line2 = [cos(theta), sin(theta), -x * cos(theta) - y * sin(theta) - l / 2];
    line3 = [-sin(theta), cos(theta), x * sin(theta) - y * cos(theta) - w / 2];
    line4 = [cos(theta), sin(theta), -x * cos(theta) - y * sin(theta) + l / 2];
    rectangle.lines = [line1;line2;line3;line4];
    rectangle.theta = theta;
end
function sum_dist = calculate_sum_dist(pts_2d, rectangle)
    pts_2d = [pts_2d ones(size(pts_2d, 1), 1)];
    lines = rectangle.lines;
    A1 = [1 0 0]; A2 = [0 1 0];
    dist_for_four = abs((lines * pts_2d')') ./ repmat(sqrt(A1 * lines' + A2 * lines'), [size(pts_2d, 1), 1]);
    [val, ind] = min(dist_for_four,[],2);
    sum_dist = sum(val);
end