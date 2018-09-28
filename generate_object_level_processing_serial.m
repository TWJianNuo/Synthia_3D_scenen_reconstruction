[base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, GT_Building_Instance_path] = get_file_storage_path();
max_frame = 294; init_ratio = 4;
to_estimation_serial_building = cell(max_frame * init_ratio); load('adjust_matrix.mat')
for frame = 1 : 1
    obj = grab_provided_data(frame);
    to_estimation_serial_building = register_building(obj, to_estimation_serial_building, frame, param_record(frame, :));
end

function obj = grab_provided_data(frame)
    [base_path, ~, ~, ~, ~, ~, GT_Building_Instance_path] = get_file_storage_path();
    f = num2str(frame, '%06d');
    ImagePath = strcat(base_path, GT_Building_Instance_path, f, '.mat'); load(ImagePath);
end
function to_estimation_serial_building = register_building(obj, to_estimation_serial_building, frame, adjust_matrix)
    for i = 1 : length(obj)
        info_block = obj{i};
        instance = info_block.instanceId;
        unit = init_unit(to_estimation_serial_building{instance});
        unit.frame(end + 1) = frame;
        unit.linear_ind{end + 1} = {info_block.linear_ind};
        unit.extrinsic_params{end + 1} = {info_block.extrinsic_params};
        unit.intrinsic_params{end + 1} = {info_block.intrinsic_params};
        unit.affine_matrx{end + 1} = {info_block.affine_matrx};
        unit.adjust_matrix{end + 1} = resh;
    end
end
function unit = init_unit(unit)
    if isempty(unit)
        unit.frame = zeros(0);
        unit.linear_ind = cell(0);
        unit.extrinsic_params = cell(0);
        unit.intrinsic_params = cell(0);
        unit.affine_matrx = cell(0);
        unit.adjust_matrix = cell(0);
    end
end