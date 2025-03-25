import pyrealsense2 as rs
import numpy as np
import cv2

#RealSense カメラの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#ストリーム開始
pipeline.start(config)

#HSV値の範囲を受け取って、その色が存在していたらtrue,していなかったらfalseを返す関数
def color(lower,upper):

    #フレームを取得
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # フレームが取得できなかった場合は False を返す
    if not color_frame:
        return False 

    #OpenCV 形式に変換
    color_image = np.asanyarray(color_frame.get_data())

    #HSV 色空間に変換
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)

    #マスクを適用
    result = cv2.bitwise_and(color_image, color_image, mask=mask)

    #物体を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50000:  #ある程度の大きさの物体を検出
                return True
    return False

try:
    while True:

        #青が存在するか判定
        if(color(np.array([80, 100, 100]),np.array([130, 255, 255]))):
            print("blue")

        #黄が存在するか判定
        if(color(np.array([15, 100, 100]),np.array([45, 255, 255]))):
            print("yellow")

        #赤が存在するか判定
        if(color(np.array([0, 100, 100]),np.array([10, 255, 255]))):
            print("red")

        #qキーが押されたら終了
        if cv2.waitKey(30) and 0xFF == ord('q'):
            break


finally:
    #ストリーム停止
    pipeline.stop()
    cv2.destroyAllWindows()
