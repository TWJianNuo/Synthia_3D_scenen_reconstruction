function union_and_save_all_marks()
    max_frame = 294;
    for i = 1 : max_frame
        org_entry = read_in_org_entry(i);
        org_entry = union_single_entry(org_entry);
        save_one_entry(org_entry, i);
    end
end
function save_one_entry(org_entry, ind)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_unioned/';
    if ind == 1
        mkdir(path);
    end
    save([path num2str(ind, '%06d') '.mat'], 'org_entry')
end
function org_entry = read_in_org_entry(frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map/';
    ind = num2str(frame, '%06d');
    loaded = load([path ind '.mat']); org_entry = loaded.prev_mark;
end
function prev_mark = union_single_entry(prev_mark)
    for i = 1 : length(prev_mark)
        to_union_ind = prev_mark{i}.instanceId; union_pool = zeros(length(prev_mark),1); 
        cur_union_pool_count = 1;
        for j = 1 : length(prev_mark)
            if to_union_ind == prev_mark{j}.instanceId
                union_pool(cur_union_pool_count) = j;
                cur_union_pool_count = cur_union_pool_count + 1;
            end
        end
        union_pool = union_pool(union_pool ~=  0);
        prev_mark = union_sub_entry(prev_mark, union_pool);
        if i >= length(prev_mark)
            break;
        end
    end
    ids = get_all_instance_ids(prev_mark);
    if length(ids) ~= length(unique(ids))
        disp('Error');
        prev_mark = zeros(0);
    end
end
function prev_mark = union_sub_entry(prev_mark, union_pool)
    new_entry = prev_mark{union_pool(1)};
    for i = 2 : length(union_pool)
        new_entry = combine_two_entry(new_entry, prev_mark{union_pool(i)});
    end
    prev_mark{union_pool(1)} = new_entry; selector = false(length(prev_mark),1); selector(union_pool(2:end)) = true;
    prev_mark(selector) = [];
end
function entry1 = combine_two_entry(entry1, entry2)
    if entry1.instanceId ~= entry2.instanceId | ~isequal(entry1.color, entry2.color)
        disp('Error'); entry1 = zeors(0);
    end
    entry1.linear_ind = [entry1.linear_ind; entry2.linear_ind];
    entry1.pts_old = [entry1.pts_old; entry2.pts_old];
    entry1.pts_new = [entry1.pts_new; entry2.pts_new];
end

function ids = get_all_instance_ids(prev_mark)
    ids = zeros(length(prev_mark),1);
    for i = 1 : length(prev_mark)
        ids(i) = prev_mark{i}.instanceId;
    end
end