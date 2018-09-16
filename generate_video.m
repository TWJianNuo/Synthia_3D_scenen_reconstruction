function generate_video()
    % base_dir = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/Car_reconstruction_results_full/';
    base_dir = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/14_Sep_2018_08_13_58_segmentation/';
    toy_object_video = VideoWriter([base_dir 'rebuilt_re']);
    toy_object_video.FrameRate = 6;
    open(toy_object_video)
    for i = 1 : 294
        % if i == 131 || i == 133
        %     continue;
        % end
        % img = imread([base_dir num2str(i, '%06d') '.png']);
        try
            img1 = imread([base_dir '3d_pts_' num2str(i, '%d') '.png']);
            img2 = imread([base_dir 'instance_label' num2str(i, '%d') '.png']);
            img1 = imresize(img1, [760, 1280]);
            img = [img2; img1];
            writeVideo(toy_object_video, img)
        catch
            continue;
        end
    end
end