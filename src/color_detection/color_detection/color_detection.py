import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2

class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection_node')

        # RealSense カメラ設定
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # 0.1秒ごとに色検出処理を実行
        self.timer = self.create_timer(0.1, self.detect_color)

    def detect_color(self):
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                return

            color_image = np.asanyarray(color_frame.get_data())
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            colors = {
                "blue": (np.array([80, 100, 100]), np.array([130, 255, 255])),
                "yellow": (np.array([15, 100, 100]), np.array([45, 255, 255])),
                "red": (np.array([0, 100, 100]), np.array([10, 255, 255]))
            }

            for colorname, (lower, upper) in colors.items():
                mask = cv2.inRange(hsv, lower, upper)
                contours,hierarchy  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50000:
                        self.get_logger().info(f'{colorname} detected')
                        break  # 一度検出したら次の色へ

        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()