function segmentation_map = get_all_instance_label(instance_label, class_label)    
    tot_type_num = 15; % in total 15 labelled categories
    segmentation_map = cell(15, 1);
    for i = 1 : tot_type_num
        segmentation_map{i}.obj_num = 0;
        segmentation_map{i}.obj = cell(segmentation_map{i}.obj_num, 1);
    end
    
    existing_instance = unique(instance_label);
    
    labelled_pixel = false(size(instance_label));
    
    for i = 1 : length(existing_instance)
        cur_instance = existing_instance(i);
        if cur_instance == 0
            continue;
        end
        [ix, iy] = find(instance_label == cur_instance);
        linear_ind = sub2ind(size(instance_label), ix, iy);
        labelled_pixel(linear_ind) = true;
        
        type = class_label(linear_ind(1));
        segmentation_map{type}.obj_num = segmentation_map{type}.obj_num + 1;
        segmentation_map{type}.obj{segmentation_map{type}.obj_num} = linear_ind;
        
        instance_label(linear_ind) = 0;
        class_label(linear_ind) = 0;
    end
    
    for i = 1 : tot_type_num
        [ix, iy] = find(class_label == i);
        linear_ind = sub2ind(size(instance_label), ix, iy);
        if ~isempty(linear_ind)
            binary_map = false(size(instance_label));
            binary_map(linear_ind) = true;
            CC = bwconncomp(binary_map);
            segmentation_map{i}.obj_num = segmentation_map{i}.obj_num + CC.NumObjects;
            for j = 1 : CC.NumObjects
                segmentation_map{i}.obj{j} = CC.PixelIdxList{j};
                class_label(CC.PixelIdxList{j}) = 0;
                instance_label(CC.PixelIdxList{j}) = 0;
            end
        end
    end
    
    if max(max(class_label)) ~= 0 | max(max(instance_label)) ~= 0
        warning('Some objects not distilled')
    end
end

