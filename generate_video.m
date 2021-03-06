function generate_video()
    base_dir = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/11_Oct_2018_21_mul/rgb_image/';
    DateString = datestr(datetime('now')); DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    toy_object_video = VideoWriter([base_dir '/' DateString]);
    toy_object_video.FrameRate = 6;
    open(toy_object_video); imsize = [760 1280]; num = 85;
    for j = 1 : num
        try
            img = imread([base_dir num2str(j) '.png']);
            % img2 = imread([base_dir2 '_instance_label' num2str(i, '%d') '.png']);
            % img3 = imread([base_dir 'align_situation_' num2str(i, '%d') '.png']);
            % img1 = imresize(img1, [760, 1280]);
            % img3 = imresize(img3, [760, 1280]);
            % img = [img2; img1];
            writeVideo(toy_object_video, img)
        catch
            continue;
        end
    end
end
function [extrinsic_params, intrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame)
    intrinsic_params = get_intrinsic_matrix();
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
    % stored_mark = load(['/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map/', f, '.mat']); stored_mark = stored_mark.prev_mark;
end