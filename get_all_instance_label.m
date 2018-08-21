function segmentation_map = get_all_instance_label(instance_label, class_label)
    % For each time, create one binary image for that catgory, then do
    % algorithm for that binary image.
    % If instance label already existing, then skip
    
    % Start on pixel with instance label
    
    tot_type_num = 15; % in total 15 labelled categories
    segmentation_map = cell(15, 1);
    for i = 1 : 15
        segmentation_map{i}.obj_num = 0;
        segmentation_map{i}.obj = cell(segmentation_map{i}.obj_num, 1);
    end
    
    existing_instance = unique(instance_label);
    
    labelled_pixel = false(size(instance_label));
    
    
    
    
    for i = 1 : length(all_instances)
        cur_instance = existing_instance(i);
        [ix, iy] = find(instance_label == cur_instance);
        linear_ind = sub2ind(size(instance_label), ix, iy);
        labelled_pixel(linear_ind) = true;
        
        type = class_label(linear_ind(1));
        segmentation_map{type}.obj_num = segmentation_map{type}.obj_num + 1;
        segmentation_map{i}.obj{segmentation_map{type}.obj_num} = linear_ind;
        
        instance_label(linear_ind) = -1;
        class_label(linear_ind) = -1;
    end
    
    
    
    for i = 1 : 15
        [ix, iy] = find(class_label == i);
        linear_ind = sub2ind(size(instance_label), ix, iy);
        if ~ISEMPTY(linear_ind)
            binary_map = false(size(instance_label));
            binary_map(linear_ind) = true;
            CC = bwconncomp(binary_map);
            segmentation_map{i}.obj_num = segmentation_map{i}.obj_num + CC.NumObjects;
            for j = 1 : CC.NumObjects
                segmentation_map{i}.obj{j} = CC.PixelIdxList{j};
                class_label(CC.PixelIdxList{j}) = -1;
                instance_label(CC.PixelIdxList{j}) = -1;
            end
        end
    end
end