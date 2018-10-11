% Helper, deal with acquiring corresponding data from different data set
function path_info = helper()
    path_info = cell(1,6);
    
    path_info{1,1} = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    path_info{1,2} = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    path_info{1,3} = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    path_info{1,4} = 'RGB/Stereo_Left/Omni_F/';
    path_info{1,5} = 'GT/COLOR/Stereo_Left/Omni_F/';
    path_info{1,6} = 'CameraParams/Stereo_Left/Omni_F/';
    path_info{1,7} = 294;
    path_info{1,8} = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYTHIA_Others/';
    %{
    path_info{3,1} = '/home/ray/Downloads/SYNTHIA-SEQS-05-SUNSET/'; % base file path
    path_info{3,2} = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    path_info{3,3} = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    path_info{3,4} = 'RGB/Stereo_Left/Omni_F/';
    path_info{3,5} = 'GT/COLOR/Stereo_Left/Omni_F/';
    path_info{3,6} = 'CameraParams/Stereo_Left/Omni_F/';
    path_info{3,7} = 707;
    path_info{3,8} = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYTHIA_Others/';
    
    path_info{2,1} = '/home/ray/Downloads/SYNTHIA-SEQS-04-SPRING/'; % base file path
    path_info{2,2} = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    path_info{2,3} = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    path_info{2,4} = 'RGB/Stereo_Left/Omni_F/';
    path_info{2,5} = 'GT/COLOR/Stereo_Left/Omni_F/';
    path_info{2,6} = 'CameraParams/Stereo_Left/Omni_F/';
    path_info{2,7} = 958;
    path_info{2,8} = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYTHIA_Others/';
    %}
end