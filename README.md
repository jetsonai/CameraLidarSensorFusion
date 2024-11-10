# CameraLidarSensorFusion

git clone https://github.com/jetsonai/CameraLidarSensorFusion

## To get datasets

https://drive.google.com/file/d/1YQAipy-mCAI_04oxECKGFkFhwjSp1L2F/view?usp=sharing

https://drive.google.com/file/d/1tUuOkCC1IvE0-mQ-aObs6IyAnrttqaUq/view?usp=sharing

cd ~/CameraLidarSensorFusion

mv ~/Downloads/datasets.tar.gz ./

tar xzf datasets.tar.gz

mv ~/Downloads/datasets.tar.gz ./

tar xzf datasets.tar.gz

-------------------------------

## camera_only

cd ~/CameraLidarSensorFusion/camera_only

python3 1_image_data_load.py ~ Q3_show_inference_viideo.py

-----------------------------------

## lidar_only

cd ~/CameraLidarSensorFusion/lidar_only

python3 1_lidar_preprocessing.py ~ 8_draw_bbox.py

-----------------------------------

## sensor_fusion

cd ~/CameraLidarSensorFusion/sensor_fusion

python3 1_translation_transform.py ~ 15_object_tracking.py

--------------------------------

## Exercise

cd ~/CameraLidarSensorFusion/quiz_solution/camera_only

python3 Q2_sensorfusion.py

python3 Q3_sensorfusion_infer_video.py

## The other exercise will be update soon !! ( in 1~2 week maybe :) )
