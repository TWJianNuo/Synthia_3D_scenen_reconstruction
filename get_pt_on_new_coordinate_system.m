function new_pts = get_pt_on_new_coordinate_system(pts)
    load('affine_matrix.mat');
    if size(pts, 2) == 3
        pts = [pts ones(size(pts, 1))];
    end
    new_pts = (affine_matrx * pts')';
end