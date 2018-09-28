[base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
intrinsic_params = get_intrinsic_matrix();
n = 294; prev_mark = cell(0); max_instance = 1; exp_re_path = make_dir();
for frame = 1 : n
    [affine_matrx, ~] = estimate_ground_plane(frame); % save('affine_matrix.mat', 'affine_matrx');
    [extrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame);
    [max_instance, prev_mark, plot_figure] = seg_image(depth, label, instance, prev_mark, extrinsic_params, intrinsic_params, affine_matrx, max_instance, frame);
    if ~isempty(plot_figure) imwrite(plot_figure, [exp_re_path '/align_situation_' num2str(frame) '.png']); end
    plot_mark(prev_mark); F = getframe(gcf); [X, ~] = frame2im(F); imwrite(X, [exp_re_path '/3d_pts_' num2str(frame) '.png']);
    new_img = render_image(prev_mark, rgb); imwrite(new_img, [exp_re_path '/_instance_label' num2str(frame) '.png']);
    save_intance(prev_mark, frame, exp_re_path)
end
save('prev_mark_record.mat', 'prev_mark_record');
function save_intance(prev_mark, frame, folder_path)
    obj = prev_mark; path = mkdir([folder_path 'Instance_map/']);
    full_path = [path num2str(frame, '%06d') '.mat'];
    save(full_path, 'obj');
end
function [extrinsic_params, depth, label, instance, rgb] = grab_provided_data(frame)
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
end
function path = make_dir()
    father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/';
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path = [father_folder DateString '_segmentation'];
    mkdir(path);
end
function new_img = render_image(prev_mark, rgb)
    r = rgb(:,:,1); g = rgb(:,:,2); b = rgb(:,:,3);
    for i = 1 : length(prev_mark)
        color = ceil(prev_mark{i}.color * 254);
        ind = prev_mark{i}.linear_ind;
        r(ind) = color(1); g(ind) = color(2); b(ind) = color(3);
    end
    new_img = uint8(zeros(size(rgb))); new_img(:,:,1) = r; new_img(:,:,2) = g; new_img(:,:,3) = b;
end
function reconstructed_3d = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
    height = size(depth_map, 1);
    width = size(depth_map, 2);
    x = 1 : height; y = 1 : width;
    [X, Y] = meshgrid(y, x);
    pts = [Y(:) X(:)];
    projects_pts = [pts(valuable_ind,2) .* depth_map(valuable_ind), pts(valuable_ind,1) .* depth_map(valuable_ind), depth_map(valuable_ind), ones(length(valuable_ind), 1)];
    reconstructed_3d = (inv(intrinsic_params * extrinsic_params) * projects_pts')';
end
function [max_instance, prev_mark, plot_figure] = seg_image(depth_map, label, instance, prev_mark, extrinsic_params, intrinsic_params, affine_matrx, max_instance, frame)
    load('adjust_matrix.mat'); plot_figure = zeros(0); obj_num_count = 1;
    if (frame > 1)  align_matrix = reshape(param_record(frame - 1, :),[4,4]); end
    building_type = 2; min_obj_pixel_num = [inf, 800, inf, inf, inf, 70, 10, 10, inf, 10, 10, inf, inf, inf, inf];
    image_size = [600 600]; SE = strel('square',3);
    [ix, iy] = find(label == building_type); linear_ind_record = sub2ind(size(depth_map), ix, iy);
    cur_old_pts_set = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, linear_ind_record);
    cur_new_pts_set = (affine_matrx * (cur_old_pts_set)')'; bimg = false(image_size);
    
    if isempty(prev_mark)
        bimg_linear_ind = ind2d(cur_new_pts_set(:, 1:2), image_size); bimg(bimg_linear_ind) = true; J = imdilate(bimg,SE); J = imerode(J,SE); CC = bwconncomp(J); cur_mark = cell(CC.NumObjects, 1);
        for i = 1 : CC.NumObjects
            now_search_b_ind = CC.PixelIdxList{i}; now_found_bindices = zeros(0);
            for j = 1 : length(now_search_b_ind)
                now_found_bindices = [now_found_bindices; find(bimg_linear_ind == now_search_b_ind(j))];
            end
            if length(now_found_bindices) > min_obj_pixel_num(building_type)
                [cur_mark{obj_num_count}, max_instance] = init_mark(max_instance, linear_ind_record(now_found_bindices), extrinsic_params, intrinsic_params, affine_matrx, cur_old_pts_set(now_found_bindices, :), rand([1 3]), cur_new_pts_set(now_found_bindices, :));
                obj_num_count = obj_num_count + 1;
            end
        end
    else
        prev_new_pts_set = zeros(0); prev_old_pts_set = zeros(0); pre_new_pts = zeros(0); integrated_instance_set = zeros(0); integrated_color_set = zeros(0);
        for i = 1 : length(prev_mark)
            prev_new_pts_set = [prev_new_pts_set; prev_mark{i}.pts_new];
            prev_old_pts_set = [prev_old_pts_set; prev_mark{i}.pts_old];
            integrated_instance_set = [integrated_instance_set; prev_mark{i}.instanceId * ones(size(prev_mark{i}.pts_old,1),1)];
            integrated_color_set = [integrated_color_set; repmat(prev_mark{i}.color, [size(prev_mark{i}.pts_old,1),1])];
        end
        integrated_instance_set = [integrated_instance_set; zeros(size(cur_old_pts_set, 1), 1)];
        trans_cur_old_pts_set = (align_matrix * cur_old_pts_set')'; trans_cur_new_pts_set =  (prev_mark{i}.affine_matrx * trans_cur_old_pts_set')';
        integrated_new_pts_set = [prev_new_pts_set; trans_cur_new_pts_set]; start_ind_of_cur_frame = size(prev_new_pts_set, 1);
        integrated_bimg_linear_ind = ind2d(integrated_new_pts_set(:, 1:2), image_size);
        prev_bimg_linear_ind = integrated_bimg_linear_ind(1 : size(prev_new_pts_set, 1)); cur_bimg_linear_ind = integrated_bimg_linear_ind(size(pre_new_pts, 1) + 1 : end);
        bimg(integrated_bimg_linear_ind) = true; J = imdilate(bimg,SE); J = imerode(J,SE); CC = bwconncomp(J); 
        cur_mark = cell(CC.NumObjects, 1);
        
        % plot_figure = generate_figure_image(trans_cur_new_pts_set, prev_new_pts_set, integrated_color_set, J);
        for i = 1 : CC.NumObjects
            now_search_b_ind = CC.PixelIdxList{i}; now_found_bindices = zeros(0);
            for j = 1 : length(now_search_b_ind)
                now_found_bindices = [now_found_bindices; find(integrated_bimg_linear_ind == now_search_b_ind(j))];
            end
            % figure(1); clf; scatter3(integrated_new_pts_set(now_found_bindices,1),integrated_new_pts_set(now_found_bindices,2),integrated_new_pts_set(now_found_bindices,3),3,integrated_color_set(now_found_bindices,:),'fill');
            if length(now_found_bindices) > min_obj_pixel_num(building_type)
                % output_split_procedure(frame, J, now_search_b_ind, prev_new_pts_set(now_found_bindices(now_found_bindices <= start_ind_of_cur_frame), :), trans_cur_new_pts_set(now_found_bindices(now_found_bindices > start_ind_of_cur_frame) - start_ind_of_cur_frame, :), integrated_color_set(now_found_bindices(now_found_bindices <= start_ind_of_cur_frame), :), i, prev_new_pts_set, trans_cur_new_pts_set)
                unique_found_instance_id = unique(integrated_instance_set(now_found_bindices));
                now_found_bindices = now_found_bindices(now_found_bindices > start_ind_of_cur_frame) - start_ind_of_cur_frame;
                if length(unique_found_instance_id) == 1 && unique_found_instance_id(1) == 0
                    [cur_mark{obj_num_count}, max_instance] = init_mark(max_instance, linear_ind_record(now_found_bindices), extrinsic_params, intrinsic_params, affine_matrx, cur_old_pts_set(now_found_bindices, :), rand([1 3]), cur_new_pts_set(now_found_bindices, :));
                else
                    father_instanceId = min(unique_found_instance_id(unique_found_instance_id~=0));
                    if (isempty(now_found_bindices)) continue; end
                    % figure(1); clf; scatter3(trans_cur_new_pts_set(now_found_bindices,1),trans_cur_new_pts_set(now_found_bindices,2),trans_cur_new_pts_set(now_found_bindices,3),3,'r','fill');
                    [cur_mark{obj_num_count}, ~] = init_mark(father_instanceId, linear_ind_record(now_found_bindices), extrinsic_params, intrinsic_params, affine_matrx, cur_old_pts_set(now_found_bindices, :), find_color(father_instanceId, prev_mark), cur_new_pts_set(now_found_bindices, :));
                end
                obj_num_count = obj_num_count + 1;
            end
        end
    end
    indices = find(~cellfun('isempty', cur_mark)); cur_mark = cur_mark(indices); 
    prev_mark = cur_mark; % plot_mark(cur_mark);
end
function output_split_procedure(frame, J, now_search_b_ind, prev_new_pts_set, trans_cur_new_pts_set, integrated_color_set, index, prev, cur)
    show_num = 28; path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/18_Sep_2018_21_segmentation/Frame28_steps/';
    if show_num == frame
        r = 255 * uint8(J); g = 255 * uint8(J); b = 255 * uint8(J);
        g(now_search_b_ind) = 0; b(now_search_b_ind) = 0;
        f = figure('visible', 'off'); clf; scatter3(prev_new_pts_set(:,1),prev_new_pts_set(:,2),prev_new_pts_set(:,3),10,integrated_color_set,'fill');
        hold on; scatter3(trans_cur_new_pts_set(:,1),trans_cur_new_pts_set(:,2),trans_cur_new_pts_set(:,3),6,'c','fill'); 
        hold on; scatter3(prev(:,1),prev(:,2),prev(:,3),3,'r','fill'); hold on; scatter3(cur(:,1),cur(:,2),cur(:,3),3,'g','fill'); axis equal;
        F = getframe(f); [X, ~] = frame2im(F); imwrite(X, [path 'split_procedure_3dpoints_' num2str(frame) '_' num2str(index) '.png'])
        color_binary_img = uint8(zeros([size(J), 3])); color_binary_img(:,:,1) = r; color_binary_img(:,:,2) = g; color_binary_img(:,:,3) = b;
        imwrite(color_binary_img, [path 'split_procedure_binary_map' num2str(frame) '_' num2str(index) '.png'])
    end
end
function X = generate_figure_image(trans_cur_new_pts_set, prev_new_pts_set, integrated_color_set, J)
    % f = figure('visible','off'); 
    f = figure(1); clf;
    clf; scatter3(trans_cur_new_pts_set(:,1),trans_cur_new_pts_set(:,2),trans_cur_new_pts_set(:,3),3,'r','fill');
    hold on; scatter3(prev_new_pts_set(:,1),prev_new_pts_set(:,2),prev_new_pts_set(:,3),5,integrated_color_set,'fill');
    figure(2); clf; imshow(fliplr(J));
    F = getframe(f); [X, ~] = frame2im(F);
end
function [cur_mark, max_instance] = init_mark(max_instance, linear_ind, extrinsic_params, intrinsic_params, affine_matrx, pts_old, color, pts_new)
    cur_mark.linear_ind = linear_ind; cur_mark.instanceId = max_instance;
    cur_mark.extrinsic_params = extrinsic_params; cur_mark.intrinsic_params = intrinsic_params;
    cur_mark.affine_matrx = affine_matrx; cur_mark.pts_old = pts_old;
    cur_mark.color = color; cur_mark.pts_new = pts_new;
    max_instance = max_instance + 1;
    
end
function color = find_color(instanceId, prev)
    color = zeros(1,3);
    for i = 1 : length(prev)
        if prev{i}.instanceId == instanceId
            color = prev{i}.color;
        end
    end
end
function plot_mark(cur_mark)
    figure('visible', 'off')
    clf;
    for i = 1 : length(cur_mark)
        pts = cur_mark{i}.pts_new;
        scatter3(pts(:,1),pts(:,2),pts(:,3),3,cur_mark{i}.color,'fill'); hold on;
    end
    axis equal
end
function bimg_linear_ind = ind2d(pts2d, image_size)
    xmin = min(pts2d(:,1)); xmax = max(pts2d(:,1)); ymin = min(pts2d(:,2)); ymax = max(pts2d(:,2));
    rangex = xmax - xmin; rangey = ymax - ymin;
    pixel_coordinate_x = (pts2d(:,1) - xmin) / (rangex / (image_size(1) - 1)); pixel_coordinate_x = round(pixel_coordinate_x) + 1;
    pixel_coordinate_y = (pts2d(:,2) - ymin) / (rangey / (image_size(2) - 1)); pixel_coordinate_y = round(pixel_coordinate_y) + 1;
    bimg_linear_ind = sub2ind(image_size, pixel_coordinate_y, pixel_coordinate_x);
end