import cv2
import numpy as np
import time 
import os.path as osp
import sys
import struct
import math
import serial
import serial.tools.list_ports
import threading

def pack_point(point):
    x, y = point
    return struct.pack('HH', int(x), int(y))

# 通信协议
STX = 0x02  # 帧起始
ETX = 0x03  # 帧结束
SEND_INTERVAL = 0.02  # 发送间隔20ms

def init_serial(port=None, baud_rate=115200, timeout=0.1):
    """初始化串口通信"""
    if port is None:
        # 自动查找可用串口
        ports = list(serial.tools.list_ports.comports())
        if len(ports) == 0:
            print("未找到可用串口")
            return None
        # 优先选择第一个串口
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
    """将偏移量格式化为带符号的4位ASCII字符串
    Args:
        value: 偏移量（-320到+320之间）
    Returns:
        4位ASCII字符串，如 "+050", "-120", "+000"
    """
    value = max(-999, min(999, value))
    return f"{value:+.3d}".replace('+', ' ') 

def send_offset_via_uart(ser, dx, dy):
    """按照指定协议发送偏移量到下位机
    帧格式: <STX> Xxxx Yyyy <ETX>
    Xxxx: X轴偏移量
    Yyyy: Y轴偏移量
    """
    if ser is None:
        return False
    try:
        # 格式化偏移量
        x_str = format_offset(dx)
        y_str = format_offset(dy)
        packet = bytes([STX]) + x_str.encode('ascii') + b' ' + y_str.encode('ascii') + bytes([ETX])
        ser.write(packet) #发送数据
        print(f"发送数据: {packet.hex()} | 偏移量: dx={dx}, dy={dy}")
        return True
    except Exception as e:
        print(f"串口发送失败: {e}")
        return False

# 全局变量用于控制线程
running = True

def raw_camera_display():
    #显示摄像头画面
    cap_raw = cv2.VideoCapture(0)
    cap_raw.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_raw.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_raw.set(cv2.CAP_PROP_FPS, 30)
    cap_raw.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # 设置摄像头参数以提高画面清晰度
    cap_raw.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  #自动曝光
    cap_raw.set(cv2.CAP_PROP_EXPOSURE, -4)  #曝光值（负数表示自动）
    cap_raw.set(cv2.CAP_PROP_AUTO_WB, 1)  #自动白平衡
    cap_raw.set(cv2.CAP_PROP_BRIGHTNESS, 128)  #亮度
    cap_raw.set(cv2.CAP_PROP_CONTRAST, 128)  #对比度
    cap_raw.set(cv2.CAP_PROP_SATURATION, 128)  #饱和度
    print("画面已启动")
    
    while running:
        ret, frame = cap_raw.read()
        if not ret:
            time.sleep(0.01)
            continue
        cv2.imshow("Raw Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            global running
            running = False
            break
    cap_raw.release()
    print("画面已退出")
    
# 摄像头标定参数（示例数值，需替换）
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([-0.3, 0.1, 0, 0], dtype=np.float32)
# 开始检测算法
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 提升 USB 传输效率

# 初始化串口（可指定端口，如 'COM3'，init_serial('COM3')，默认自动查找）
ser = init_serial() 
# 画面中心点
IMAGE_CENTER = (320, 240)
prev_time = time.time()
fps = 0
last_send_time = time.time()  # 上次发送时间
step_per_frame = 1      # 每帧只描绘1个点
frame_count = 0  

# 启动投屏线程
raw_display_thread = threading.Thread(target=raw_camera_display, daemon=True)
raw_display_thread.start()

while running:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # 转换为HSV颜色空间，用于检测激光点
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 红色激光颜色范围
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # 对掩码进行形态学操作，去除噪点
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 查找激光点轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的激光点
    max_area = 0
    laser_center = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            # 计算轮廓中心
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                laser_center = (cX, cY)
    
    if laser_center is not None:
        # 在原图上标记激光点
        cv2.circle(frame, laser_center, 8, (0, 0, 255), -1)
        cv2.putText(frame, "Laser Point", (laser_center[0] + 10, laser_center[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        dx = laser_center[0] - IMAGE_CENTER[0]  # 水平偏移量
        dy = laser_center[1] - IMAGE_CENTER[1]  # 垂直偏移量
        print(f"检测到的激光点：({laser_center[0]}, {laser_center[1]}) | 偏移量: dx={dx}, dy={dy}")
        
        # 定时返回数据
        current_time = time.time()
        if current_time - last_send_time >= SEND_INTERVAL:
            send_offset_via_uart(ser, dx, dy)
            last_send_time = current_time
    else:
        cv2.putText(frame, "未检测到激光点", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 未检测到激光点时也定时发送零偏移
        current_time = time.time()
        if current_time - last_send_time >= SEND_INTERVAL:
            send_offset_via_uart(ser, 0, 0)
            last_send_time = current_time
    
    # 显示处理结果
    cv2.imshow("Detected Frame", frame)
    cv2.imshow("Laser Mask", mask)

    # 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# 等待线程结束
raw_display_thread.join(timeout=1)
cap.release()
cv2.destroyAllWindows()
# 关闭串口
if ser is not None and ser.is_open:
    ser.close()
    print("串口已关闭")