function pts = sample_cubic_by_num(cuboid, num1, num2)
    % max = 0.9; 
    % min = 0.1;
    max = 0.9;
    min = 0.1;
    k1 = min : (max - min)/(num1 - 1) : max;
    k2 = min : (max - min)/(num2 - 1) : max;
    pts = zeros(4 * num1 * num2, 6); % 3D location plus belongings, last two for k1 and k2
    for i = 1 : 4
        x = k1 * cuboid{i}.length1 * cuboid{i}.plane_dirs(1, 1) + cuboid{i}.pts(1, 1);
        y = k1 * cuboid{i}.length1 * cuboid{i}.plane_dirs(1, 2) + cuboid{i}.pts(1, 2);
        z = k2 * cuboid{i}.length2;        
        [xx, zz] = meshgrid(x, z);
        [yy, zz] = meshgrid(y, z);
        [kk1, kk2] = meshgrid(k1, k2);
        pts((i - 1) * num1 * num2 + 1 : i * num1 * num2, :) = [xx(:) yy(:) zz(:) ones(length(zz(:)), 1) * i kk1(:) kk2(:)];
    end
end