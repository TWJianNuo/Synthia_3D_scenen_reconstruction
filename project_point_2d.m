function [pts, depth] = project_point_2d(extrinsic, intrinsic, pts, affine)
    if nargin == 4
        extrinsic = extrinsic * inv(affine);
    end
    if size(pts, 2) < 4
        pts = [pts ones(size(pts,1), 1)];
    end
    pts = (intrinsic * extrinsic * pts')'; pts(:,1) = pts(:,1) ./ pts(:,3); pts(:,2) = pts(:,2) ./ pts(:,3); 
    depth = pts(:, 3); pts = pts(:, 1:2);
end