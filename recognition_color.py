import pyrealsense2 as rs
import numpy as np
import cv2

#RealSense カメラの設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#ストリーム開始
pipeline.start(config)

try:
    while True:
        #フレームを取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        #OpenCV 形式に変換
        color_image = np.asanyarray(color_frame.get_data())

        #HSV 色空間に変換
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        #青色の範囲を定義
        lower_blue = np.array([80, 100, 100])   #青の下限
        upper_blue = np.array([130, 255, 255])  #青の上限
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        #黄の範囲を定義
        lower_yellow = np.array([15, 100, 100])   #黄の下限
        upper_yellow = np.array([45, 255, 255])  #黄の上限
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        #赤の範囲を定義
        lower_red = np.array([160, 80, 80])   #赤の下限
        upper_red = np.array([179, 255, 255])  #赤の上限
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        #マスクを適用
        blue_result = cv2.bitwise_and(color_image, color_image, mask=blue_mask)
        yellow_result = cv2.bitwise_and(color_image, color_image, mask=yellow_mask)
        red_result = cv2.bitwise_and(color_image, color_image, mask=red_mask)

        #青色の物体を検出
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #黄色の物体を検出
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #赤色の物体を検出
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in blue_contours:
            area = cv2.contourArea(contour)
            if area > 5000:  #ある程度の大きさの青い物体を検出
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(color_image, "blue", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                print("青")

        for contour in yellow_contours:
            area = cv2.contourArea(contour)
            if area > 5000:  #ある程度の大きさの黄色の物体を検出
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(color_image, "yellow", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                print("黄")

        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 5000:  #ある程度の大きさの赤い物体を検出
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(color_image, "red", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("赤")

        #画像を表示
        cv2.imshow('RealSense Color', color_image)

        #'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    #ストリーム停止
    pipeline.stop()
    cv2.destroyAllWindows()
