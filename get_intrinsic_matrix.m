function intrinsic_params = get_intrinsic_matrix()
    focal = 532.7403520000000; cx = 640; cy = 380;
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
end

