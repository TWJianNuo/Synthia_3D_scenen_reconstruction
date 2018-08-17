function depthMatrix = getDepth(ImagePath)
% read the image
im = imread(ImagePath);

% r -> Red channel
% g -> Green channel
% b -> Blue channel
% f -> Far: the max render distance. 1000 because we want meters.

% depth = ((r) + (g * 256) + (b * 256*256)) / ((256*256*256) - 1) * f

depthMatrix = double(im(:,:,1)) / 1000;
end