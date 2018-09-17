function [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, GT_Building_Instance_path] = get_file_storage_path()
    base_path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    GT_Depth_path = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    GT_seg_path = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    GT_RGB_path = 'RGB/Stereo_Left/Omni_F/';
    GT_Color_Label_path = 'GT/COLOR/Stereo_Left/Omni_F/';
    GT_Building_Instance_path = 'GT/INSTANCE_BUILDINGS/Stereo_Left/Omni_F';
    cam_para_path = 'CameraParams/Stereo_Left/Omni_F/';
end

