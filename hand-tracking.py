"""
手部追踪云台 — 双线程稳定版
===================================
- 主线程: 读取摄像头 + 实时投屏显示（零处理开销）
- 识别线程: 独立读取摄像头 + 每4秒1帧手部检测
- 串口输出: 手部相对画面中心的偏移量 → 下位机控制云台

目标平台: 香橙派3B (4GB RAM) / 树莓派 / 通用 Linux+ARM
"""

import cv2
import mediapipe as mp
import threading
import time
import queue
import sys
import os

if sys.platform.startswith('linux') and 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

# ============================================================
# 配置区
# ============================================================

CAMERA_INDEX = 0
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
DETECT_WIDTH = 480
DETECT_HEIGHT = 360
DETECT_INTERVAL = 1.0

MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MAX_NUM_HANDS = 1

SERIAL_ENABLED = True
SERIAL_PORT = '/dev/ttyS2'
SERIAL_BAUDRATE = 9600

WINDOW_NAME = 'Camera Preview'
SHOW_FPS = True
DRAW_BOX = True


# ============================================================
# 串口发送器
# ============================================================
class SerialSender:
    def __init__(self, port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._connect()

    def _connect(self):
        try:
            import serial
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
            # 禁止 DTR/RTS 复位信号，避免下位机一开串口就重启
            self.ser.dtr = False
            self.ser.rts = False
            time.sleep(0.5)  # 等下位机稳定
            print(f'[串口] 已连接 {self.port} @ {self.baudrate}')
        except ImportError:
            print('[串口] pyserial 未安装')
            self.ser = None
        except Exception as e:
            print(f'[串口] 连接失败 {self.port}: {e}')
            self.ser = None

    def send(self, hand_x: float, hand_y: float):
        if self.ser is None:
            return
        try:
            # 添加负号反转方向，解决手在左边摄像头向右转的问题
            offset_x = -(hand_x - 0.5)
            offset_y = -(hand_y - 0.5)
            msg = f'{offset_x:.4f},{offset_y:.4f}/n'
            n = self.ser.write(msg.encode('utf-8'))
            print(f'{offset_x:.4f},{offset_y:.4f} (已发送 {n} 字节)')
        except Exception as e:
            print(f'[串口发送失败] {e}')

    def close(self):
        if self.ser:
            self.ser.close()


# ============================================================
# 手部识别器
# ============================================================
class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        print(f'[识别器] MediaPipe Hands 初始化完成')

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, None, 0.0, None

        landmarks = results.multi_hand_landmarks[0]
        confidence = 1.0
        if results.multi_handedness:
            confidence = results.multi_handedness[0].classification[0].score

        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]

        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0

        bbox_px = {
            'x_min': int(min(xs) * w),
            'y_min': int(min(ys) * h),
            'x_max': int(max(xs) * w),
            'y_max': int(max(ys) * h),
            'cx_px': int(cx * w),
            'cy_px': int(cy * h),
        }
        return cx, cy, confidence, bbox_px

    def close(self):
        self.hands.close()


# ============================================================
# 共享状态
# ============================================================
class SharedState:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.latest_result = {
            'cx': None, 'cy': None, 'confidence': 0.0, 'bbox': None, 'found': False,
        }
        self.lock = threading.Lock()


# ============================================================
# 识别线程（从队列获取帧）
# ============================================================
def recognition_thread_func(detector, state, serial_sender):
    time.sleep(1.0)
    print(f'[识别线程] 启动，检测间隔={DETECT_INTERVAL}s')

    last_detect_time = time.time() - DETECT_INTERVAL

    while state.running:
        frame = None
        try:
            frame = state.frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        current_time = time.time()
        if current_time - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = current_time

            frame_small = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT),
                                    interpolation=cv2.INTER_NEAREST)
            cx, cy, conf, bbox = detector.detect(frame_small)

            with state.lock:
                if cx is not None:
                    state.latest_result['cx'] = cx
                    state.latest_result['cy'] = cy
                    state.latest_result['confidence'] = conf
                    state.latest_result['bbox'] = bbox
                    state.latest_result['found'] = True
                else:
                    state.latest_result['found'] = False
                    state.latest_result['bbox'] = None

                if serial_sender:
                    if cx is not None:
                        serial_sender.send(cx, cy)
                    else:
                        serial_sender.send(0.5, 0.5)

        time.sleep(0.01)

    print('[识别线程] 退出')


# ============================================================
# 主函数
# ============================================================
def main():
    print('=' * 55)
    print('  手部追踪云台 — 双线程优化版')
    print(f'  投屏分辨率: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} (高画质)')
    print(f'  识别分辨率: {DETECT_WIDTH}x{DETECT_HEIGHT} (极限低)')
    print(f'  识别间隔:   {DETECT_INTERVAL}s/帧')
    print(f'  模型复杂度: {MODEL_COMPLEXITY} (轻量)')
    print('  按 Q 键退出程序')
    print('=' * 55)

    # ---- 摄像头初始化（单摄像头 + 帧队列共享）----
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print('[错误] 无法打开摄像头！请检查:')
        print('  1. 摄像头是否连接')
        print('  2. CAMERA_INDEX 是否正确（尝试 0, 1, 2...）')
        sys.exit(1)

    # ---- 设置摄像头属性 ----
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[摄像头] 实际分辨率: {actual_w}x{actual_h}')

    # ---- 初始化手部检测器 ----
    detector = HandDetector()

    # ---- 初始化串口（如启用）----
    serial_sender = None
    if SERIAL_ENABLED:
        serial_sender = SerialSender()

    # ---- 共享状态 ----
    state = SharedState()

    # ---- 启动识别线程 ----
    recog_thread = threading.Thread(
        target=recognition_thread_func,
        args=(detector, state, serial_sender),
        daemon=True
    )
    recog_thread.start()

    # ---- 主循环：投屏显示（零处理）----
    fps_counter = 0
    fps_timer = time.time()
    display_fps = 0

    print('\n[主线程] 投屏窗口已启动')
    print('[提示] 按 Q 键退出\n')

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # 顺时针旋转90度
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # 帧放入队列供识别线程使用
        try:
            if state.frame_queue.full():
                try:
                    state.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            state.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

        if DRAW_BOX:
            with state.lock:
                bbox = state.latest_result.get('bbox')
                found = state.latest_result.get('found', False)
                cx_norm = state.latest_result.get('cx')
                cy_norm = state.latest_result.get('cy')
                conf = state.latest_result.get('confidence', 0.0)

            if found and bbox is not None:
                scale_x = actual_w / DETECT_WIDTH
                scale_y = actual_h / DETECT_HEIGHT
                x1 = int(bbox['x_min'] * scale_x)
                y1 = int(bbox['y_min'] * scale_y)
                x2 = int(bbox['x_max'] * scale_x)
                y2 = int(bbox['y_max'] * scale_y)
                cxp = int(bbox['cx_px'] * scale_x)
                cyp = int(bbox['cy_px'] * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.drawMarker(frame, (cxp, cyp), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                if cx_norm is not None:
                    offset_x = cx_norm - 0.5
                    offset_y = cy_norm - 0.5
                    cv2.putText(frame, f'({offset_x:+.2f}, {offset_y:+.2f})',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if SHOW_FPS:
            fps_counter += 1
            if fps_counter >= 30:
                now = time.time()
                display_fps = 30.0 / (now - fps_timer)
                fps_timer = now
                fps_counter = 0
            cv2.putText(frame, f'FPS: {display_fps:.1f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    print('[清理] 释放资源...')
    state.running = False
    recog_thread.join(timeout=2.0)
    cap.release()
    detector.close()
    if serial_sender:
        serial_sender.close()
    cv2.destroyAllWindows()
    print('[退出] 程序结束')


if __name__ == '__main__':
    main()
