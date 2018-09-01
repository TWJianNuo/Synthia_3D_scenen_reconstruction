function Generate_video()
    base_dir = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/Car_reconstruction_results/';
    toy_object_video = VideoWriter([base_dir 'car_re']);
    toy_object_video.FrameRate = 6;
    open(toy_object_video)
    for i = 1 : 294
        if i == 131 || i == 133
            continue;
        end
        img = imread([base_dir num2str(i, '%06d') '.png']);
        writeVideo(toy_object_video, img)
    end
end