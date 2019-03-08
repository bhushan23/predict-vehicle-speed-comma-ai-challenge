import cv2

def convert_from_video_to_images(input_vid, output_dir):
    vidcap = cv2.VideoCapture(input_vid)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print('Frames per sec: ', fps)
    success, image = vidcap.read()
    count = 0
    while success:
        img_frame = output_dir + "frame_" + str(count) + ".jpg"
        cv2.imwrite(img_frame, image)     # save frame as JPEG file      
        success, image = vidcap.read()
        if count % 500 == 0:
            print('Converting: ', count)
        count += 1
    print(count, " frames converted")

data_path = "/home/bhushan/work/full_time/comma/data/"
out_path  = "/home/bhushan/work/full_time/comma/my_work/predict-vehicle-speed-comma-ai-challenge/data/"

convert_from_video_to_images(data_path + 'train.mp4', out_path + 'train/')
convert_from_video_to_images(data_path + 'test.mp4', out_path + 'test/')