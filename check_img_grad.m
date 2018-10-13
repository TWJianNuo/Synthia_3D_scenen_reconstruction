depth_map = imread('/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/Depth/Stereo_Left/Omni_F/000001.png'); 
depth_map = double(rgb2gray(depth_map));
repeated_time = 100; 
num_it = 500; step_size = 1; delta = 1;
for frame = 1 : repeated_time
    y_record = zeros(num_it, 1);
    pts = double([randi([1 size(depth_map,2)], 1), randi([1 size(depth_map, 1)], 1)]);
    for i = 1 : num_it
        pts_rounded = round(pts);
        grad = (image_grad_(depth_map, pts_rounded, step_size)); 
        pts = pts - grad * delta; pts_rounded = round(pts);
        try
            y_record(i) = depth_map(pts_rounded(2), pts_rounded(1));
        catch
            break;
        end
    end
    figure(1); clf; stem(y_record); F = getframe(gcf); [X, Map] = frame2im(F);
    imwrite(X, ['/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/' num2str(frame) '.png'])
end
function grad = image_grad_(image, location, step_size)
    x_grad = interpImg(image, [location(1) + step_size, location(2)]) - interpImg(image, [location(1), location(2)]);
    y_grad = interpImg(image, [location(1), location(2) + step_size]) - interpImg(image, [location(1), location(2)]);
    grad = [x_grad y_grad];
end