function gt_Matrix = getIDs(ImagePath)
        % read the image
        im = imread(ImagePath);
        im = double(im);

        labelMatrix = im(:,:,1);
        instanceMatrix = im(:,:,2);
        
        gt_Matrix = struct;
        gt_Matrix.label = labelMatrix;
        gt_Matrix.instance = instanceMatrix;
    end