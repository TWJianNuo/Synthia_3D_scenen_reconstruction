function generate_introduction_statement()
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/Car_reconstruction_results/help.txt';
    fileId = fopen(path, 'w');
    statement = {
        'camera_info.txt: intrinsic matrix, extrinsic matrix, affine matrix.\n';
        'is_frame_valid.txt: whether this frame is considered (a frame whose ground plane not flat is discarded.\n';
        'cubic_info.txt: cx, cy, theta, l, w, h.\n';
        'metric.txt: Distance of points to cubic shape, differences in depth.\n';
        };
    for i = 1 : length(statement)
        fprintf(fileId, statement{i});
    end
    fclose(fileId);
end

