function [label, instance] = getIDs(ImagePath)
        % read the image
        im = imread(ImagePath);
        im = double(im);

        labelMatrix = im(:,:,1);
        instanceMatrix = im(:,:,2);

        label = labelMatrix;
        instance = instanceMatrix;
    end