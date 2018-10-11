function cuboid = generate_cuboid(x, y, theta, l, w, h)
    base_dir = [1 0 0];
    R = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
    R_90 = [0 -1 0; 1 0 0; 0 0 1];
    base_dirs = zeros(4, 3);
    base_dirs(1, :) = (R * base_dir')';
    base_dirs(2, :) = (R_90 * base_dirs(1, :)')';
    base_dirs(3, :) = (R_90 * base_dirs(2, :)')';
    base_dirs(4, :) = (R_90 * base_dirs(3, :)')';
    plane_dirs = [base_dirs(2, :);base_dirs(3, :); base_dirs(4, :); base_dirs(1, :)];
    
    cuboid = cell([6 1]);
    length = [l w l w];
    cuboid{1}.theta = theta;
    for i = 1 : 4
        cuboid{i}.pts = zeros(4, 3);
        if i == 1
            cuboid{i}.pts(1, :) = [x y 0];
        else
            cuboid{i}.pts(1, :) = cuboid{i - 1}.pts(2, :);
        end
        cuboid{i}.pts(2, :) = cuboid{i}.pts(1, :) + base_dirs(i, :) * length(i);
        cuboid{i}.pts(3, :) = cuboid{i}.pts(2, :) + [0 0 h];
        cuboid{i}.pts(4, :) = cuboid{i}.pts(1, :) + [0 0 h];
        cuboid{i}.dir = plane_dirs(i, :);
        cuboid{i}.length1 = length(i);
        cuboid{i}.length2 = h;
        cuboid{i}.params = [cuboid{i}.dir -(cuboid{i}.dir * cuboid{i}.pts(1, :)')];
        cuboid{i}.plane_dirs = [base_dirs(i, :); [0 0 1]];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dir1 = cuboid{i}.plane_dirs(1, :);
        dir2 = cuboid{i}.dir;
        dir3 = [0 0 1];
        new_dirs = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]';
        old_dirs = [[dir1 0]; [dir2 0]; [dir3 0]; [0 0 0 1]];
        cuboid{i}.T = new_dirs' * inv(old_dirs');
        Transition_matrix = -[cuboid{i}.pts(1, :) * dir1' cuboid{i}.pts(1, :) * dir2' cuboid{i}.pts(1, :) * dir3'];
        cuboid{i}.T(1:3, 4) = Transition_matrix';
        
        dir1 = cuboid{i}.plane_dirs(1, :);
        dir2 = [0 0 -1];
        dir3 = cross(dir1(1:3), dir2(1:3));
        new_dirs = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]';
        old_dirs = [[dir1 0]; [dir2 0]; [dir3 0]; [0 0 0 1]];
        cuboid{i}.Tflat = new_dirs' * inv(old_dirs');
        Transition_matrix = -[cuboid{i}.pts(1, :) * dir1' cuboid{i}.pts(1, :) * dir2' cuboid{i}.pts(1, :) * dir3'];
        cuboid{i}.Tflat(1:3, 4) = Transition_matrix';
    end
    i = 5;
    if i == 5
        cuboid{i}.pts = [
            cuboid{1}.pts(1, :) + [0 0 h];
            cuboid{2}.pts(1, :) + [0 0 h];
            cuboid{3}.pts(1, :) + [0 0 h];
            cuboid{4}.pts(1, :) + [0 0 h];
            ];
        cuboid{i}.dir = [0 0 -1];
        cuboid{i}.length1 = l;
        cuboid{i}.length2 = w;
        cuboid{i}.params = [cuboid{i}.dir -(cuboid{i}.dir * cuboid{i}.pts(1, :)')];
        cuboid{i}.plane_dirs = [base_dirs(1, :); base_dirs(2, :)];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dir1 = cuboid{i}.plane_dirs(1, :);
        dir2 = cuboid{i}.plane_dirs(2, :);
        dir3 = [0 0 -1];
        new_dirs = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]';
        old_dirs = [[dir1 0]; [dir2 0]; [dir3 0]; [0 0 0 1]];
        cuboid{i}.T = new_dirs' * inv(old_dirs');
        Transition_matrix = -[cuboid{i}.pts(1, :) * dir1' cuboid{i}.pts(1, :) * dir2' cuboid{i}.pts(1, :) * dir3'];
        cuboid{i}.T(1:3, 4) = Transition_matrix';
    end
    i = 6;
    if i ~= 6
        cuboid{i}.pts = [
            cuboid{1}.pts(1, :);
            cuboid{2}.pts(1, :);
            cuboid{3}.pts(1, :);
            cuboid{4}.pts(1, :);
            ];
        cuboid{i}.dir = [0 0 1];
        cuboid{i}.length1 = l;
        cuboid{i}.length2 = w;
        cuboid{i}.params = [cuboid{i}.dir -(cuboid{i}.dir * cuboid{i}.pts(1, :)')];
        cuboid{i}.plane_dirs = [base_dirs(1, :); base_dirs(2, :)];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        dir1 = cuboid{i}.plane_dirs(1, :);
        dir2 = cuboid{i}.plane_dirs(2, :);
        dir3 = [0 0 1];
        new_dirs = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]';
        old_dirs = [[dir1 0]; [dir2 0]; [dir3 0]; [0 0 0 1]];
        cuboid{i}.T = new_dirs' * inv(old_dirs');
        Transition_matrix = -[cuboid{i}.pts(1, :) * dir1' cuboid{i}.pts(1, :) * dir2' cuboid{i}.pts(1, :) * dir3'];
        cuboid{i}.T(1:3, 4) = Transition_matrix';
    end
end