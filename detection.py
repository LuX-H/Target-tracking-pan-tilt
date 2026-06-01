"""
激光点检测
  - 检测线程：摄像头采集 → 激光点检测 → 串口发送偏移量
  - 主线程：投屏原始画面 + 检测标注画面 + 遮罩画面
"""

import cv2
import numpy as np
import time
import threading
import serial
import serial.tools.list_ports

# ==================== 通信协议 ====================
STX = 0x02
ETX = 0x03
SEND_INTERVAL = 0.02  # 50Hz

# ==================== 串口通信 ====================
def init_serial(port=None, baud_rate=115200, timeout=0.1):
    """初始化串口，未指定端口则自动查找"""
    if port is None:
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("未找到可用串口")
            return None
        port = ports[0].device
        print(f"自动选择串口: {port}")

    try:
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        print(f"串口 {port} 已打开，波特率: {baud_rate}")
        return ser
    except Exception as e:
        print(f"打开串口失败: {e}")
        return None


def format_offset(value):
    """将偏移量格式化为4位带符号ASCII，范围[-999, 999]"""
    value = max(-999, min(999, value))
    return f"{value:+.3d}".replace('+', ' ')


def send_offset_via_uart(ser, dx, dy):
    """按协议帧格式发送：<STX> Xxxx Yyyy <ETX>"""
    if ser is None or not ser.is_open:
        return False
    try:
        x_str = format_offset(dx)
        y_str = format_offset(dy)
        packet = bytes([STX]) + x_str.encode() + b' ' + y_str.encode() + bytes([ETX])
        ser.write(packet)
        return True
    except Exception as e:
        print(f"串口发送失败: {e}")
        return False


# ==================== 激光点检测 ====================
LOWER_RED1 = np.array([0, 80, 60])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 80, 60])
UPPER_RED2 = np.array([180, 255, 255])
MORPH_KERNEL = np.ones((3, 3), np.uint8)
BLUR_SIZE = (15, 15)

def create_red_mask(hsv):
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    return cv2.bitwise_or(mask1, mask2)


def detect_laser(frame):
    """
    检测红色激光点。
    利用干涉条纹对称分布的特性，对红色遮罩做高斯模糊后取峰值位置，
    对称噪声不偏移质心，天然抗干涉环干扰。
    Returns:
        (x, y) 或 None, mask
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = create_red_mask(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)

    if cv2.countNonZero(mask) < 15:
        return None, mask
    blurred = cv2.GaussianBlur(mask.astype(np.float32), BLUR_SIZE, 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)

    if blurred[max_loc[1], max_loc[0]] < 1.0:
        return None, mask
    return max_loc, mask


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.raw_frame = None          # 原始画面
        self.annotated_frame = None    # 检测标注画面
        self.mask = None               # 红色遮罩
        self.has_laser = False
        self.dx = 0
        self.dy = 0
        self.fps = 0.0
        self.running = True

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self):
        with self._lock:
            return (
                self.raw_frame.copy()           if self.raw_frame is not None          else None,
                self.annotated_frame.copy()     if self.annotated_frame is not None    else None,
                self.mask.copy()                if self.mask is not None               else None,
                self.has_laser,
                self.dx,
                self.dy,
                self.fps,
                self.running,
            )
    def stop(self):
        with self._lock:
            self.running = False

    def is_running(self):
        with self._lock:
            return self.running

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_CENTER = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

def detection_thread(cap, ser, shared):
    """采集、检测、串口发送"""
    prev_time = time.time()
    last_send_time = time.time()

    while shared.is_running():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        # FPS
        now = time.time()
        dt = now - prev_time
        fps = 1.0 / dt if dt > 0.001 else 0
        prev_time = now
        raw_frame = frame.copy()
        laser_center, mask = detect_laser(frame)
        annotated = frame.copy()
        dx, dy = 0, 0
        has_laser = False

        if laser_center is not None:
            dx = laser_center[0] - IMAGE_CENTER[0]
            dy = laser_center[1] - IMAGE_CENTER[1]
            has_laser = True

            # 绘制检测标注
            cv2.circle(annotated, laser_center, 8, (0, 255, 0), -1)
            cv2.circle(annotated, IMAGE_CENTER, 4, (255, 0, 0), -1)
            cv2.line(annotated, IMAGE_CENTER, laser_center, (0, 255, 255), 2)
            cv2.putText(annotated, f"dx={dx:+.3d} dy={dy:+.3d}",
                        (laser_center[0] + 12, laser_center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 按间隔发送
            if now - last_send_time >= SEND_INTERVAL:
                if send_offset_via_uart(ser, dx, dy):
                    print(f"发送: dx={dx:+.3d}, dy={dy:+.3d}")
                last_send_time = now
        else:
            cv2.putText(annotated, "No Laser", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, FRAME_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 写入共享状态
        shared.update(
            raw_frame=raw_frame,
            annotated_frame=annotated,
            mask=mask,
            has_laser=has_laser,
            dx=dx, dy=dy,
            fps=fps,
        )

def main():
    # 摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    ser = init_serial()
    shared = SharedState()
    det_thread = threading.Thread(
        target=detection_thread,
        args=(cap, ser, shared),
        name="Detection",
        daemon=True,
    )

    det_thread.start()
    print("激光点检测已启动 | 按 'q' 退出")
    cv2.namedWindow("Raw Camera Feed", cv2.WINDOW_NORMAL)      # 投屏窗口
    cv2.namedWindow("Laser Detection", cv2.WINDOW_NORMAL)      # 检测标注
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)                 # 红色遮罩

    # 主循环
    while True:
        raw_frame, annotated, mask, has_laser, dx, dy, fps, running = shared.snapshot()

        if not running:
            break
        if raw_frame is not None:
            cv2.imshow("Raw Camera Feed", raw_frame)
        if annotated is not None:
            cv2.imshow("Laser Detection", annotated)
        if mask is not None:
            cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared.stop()
            break

    det_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.close()
        print("串口已关闭")
    print("程序已退出")

if __name__ == '__main__':
    main()