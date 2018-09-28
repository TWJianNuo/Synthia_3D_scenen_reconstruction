function judge_exp_re()
    base_dir = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/28_Sep_2018_08/';
    DateString = datestr(datetime('now')); DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    toy_object_video = VideoWriter([base_dir DateString]);
    toy_object_video.FrameRate = 6;
    open(toy_object_video); it_num = 100; size = [843 1438];
    [delta_record, params_record, final_params_record, gt_params_record] = read_metric(base_dir);
    correct_count = 0;
    for i = 1 : it_num
        try
            [img2d_s, img3d_s] = read_start(i, base_dir, size);
            [img2d_e, img3d_e] = read_end(i, base_dir, size);
            [error_img, padding] = read_error(i, base_dir, size);
            re = judge_sucess(final_params_record(i, :), gt_params_record(i,:), img3d_e); 
            if re
                correct_count = correct_count + 1;
            end
            padding = process_img(padding, re);
            img = [img2d_s img3d_s; img2d_e img3d_e; error_img padding];
            writeVideo(toy_object_video, img)
        catch
            continue
        end
    end
    correct_rate = correct_count / it_num
end
function padding = process_img(padding, is_right)
    im_size = size(padding); im_size = im_size(1:2);
    if ~is_right
        color = 'red';
    else
        color = 'gree';
    end
    padding = insertShape(uint8(padding), 'FilledCircle', [im_size(2)/2 im_size(1)/2 50], 'Color', color);
end
function re = judge_sucess(final_params_record, gt_params_record, img3d_e)
    re = false; 
    ratioth = 0.3;
    % figure(1); clf; imshow(img3d_e)
    % abs(final_params_record(4:5) - gt_params_record(4:5)) ./ abs(gt_params_record(4:5))
    if max(abs(final_params_record(4:5) - gt_params_record(4:5)) ./ abs(gt_params_record(4:5))) < ratioth && abs(final_params_record(1) - gt_params_record(1)) ./ abs(gt_params_record(1)) < 0.1
        re = true;
    end
end
function [delta_record, params_record, final_params_record, gt_params_record] = read_metric(path)
    tot_num = 100;
    f1 = fopen([path 'delta_record.txt'],'r');
    delta_record = cell2mat(textscan(f1,'%f\t%f\n'));
    f2 = fopen([path 'params_diff.txt'],'r');
    params_record = cell2mat(textscan(f2,'%f\t%f\t%f\t%f\t%f\t%f\n'));
    f3 = fopen([path 'final_params_record.txt'],'r');
    final_params_record = cell2mat(textscan(f3,'%f\t%f\t%f\t%f\t%f\t%f\n'));
    f4 = fopen([path 'gt_params_record.txt'],'r');
    gt_params_record = cell2mat(textscan(f4,'%f\t%f\t%f\t%f\t%f\t%f\n'));
end
function [error_img, padding] = read_error(ind, path, size)
    error_img = imread([path num2str(ind) '_error.png']); error_img = imresize(error_img, size);
    padding = int8(zeros([size 3]));
end
function [img2d, img3d] = read_start(ind, path, size)
    img2d = imread([path num2str(ind) '_2d_img_' num2str(1) '.png']); img2d = imresize(img2d, size);
    img3d = imread([path num2str(ind) '_3d_img_' num2str(1) '.png']); img3d = imresize(img3d, size);
end
function [img2d, img3d] = read_end(ind, path, size)
    max_it = 200;
    for i = 1 : max_it
        try
            img2d = imread([path num2str(ind) '_2d_img_' num2str(i) '.png']); 
            img3d = imread([path num2str(ind) '_3d_img_' num2str(i) '.png']); 
        catch
        end
    end
    img2d = imresize(img2d, size); img3d = imresize(img3d, size);
end