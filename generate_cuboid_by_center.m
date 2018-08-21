function cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h)
    base_dir = [1 0 0];
    R = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
    R_90 = [0 -1 0; 1 0 0; 0 0 1];
    base_dirs = zeros(4, 3);
    base_dirs(1, :) = (R * base_dir')';
    base_dirs(2, :) = (R_90 * base_dirs(1, :)')';
    base_dirs(3, :) = (R_90 * base_dirs(2, :)')';
    base_dirs(4, :) = (R_90 * base_dirs(3, :)')';
    m = [cx, cy, 0] - base_dirs(1, :) * l / 2 - base_dirs(2, :) *w / 2;
    x = m(1);
    y = m(2);
    cuboid = generate_cuboid(x, y, theta, l, w, h);
end