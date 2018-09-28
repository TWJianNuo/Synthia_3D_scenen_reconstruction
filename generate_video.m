function generate_video()
    base_dir = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/26_Sep_2018_19';
    DateString = datestr(datetime('now')); DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    toy_object_video = VideoWriter([base_dir '/' DateString]);
    toy_object_video.FrameRate = 6;
    open(toy_object_video)
    for i = 1 : 4000
        try
            img1 = imread([base_dir '/2d_img' num2str(i) '.png']);
            img2 = imread([base_dir '/3d_img' num2str(i) '.png']);
            img2 = imresize(img2, [size(img1, 1) size(img1, 2)]); img = [img1 img2];
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
    for i = 1 : 10
        img_ = imread([base_dir '/error' '.png']);
        img_ = imresize(img_, [size(img, 1) size(img, 2)]);
        writeVideo(toy_object_video,img_)
    end
end