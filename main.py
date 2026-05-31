#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import serial
import sys

class LaserTracker:
    def __init__(self, serial_port='/dev/ttyS1', baud_rate=9600):
        # 摄像头初始化
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("错误：无法打开摄像头")
            sys.exit(1)
        
        # 获取摄像头分辨率
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头分辨率: {self.width} x {self.height}")
        
        # 画面中心点
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        print(f"画面中心点: ({self.center_x}, {self.center_y})")
        
        # 串口初始化
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            print(f"串口已打开: {serial_port} @ {baud_rate}bps")
        except serial.SerialException as e:
            print(f"错误：无法打开串口 {serial_port}: {e}")
            self.ser = None
        
        # 激光点检测参数（针对红色激光优化）
        self.laser_lower_red1 = np.array([0, 120, 180])
        self.laser_upper_red1 = np.array([8, 255, 255])
        self.laser_lower_red2 = np.array([172, 120, 180])
        self.laser_upper_red2 = np.array([180, 255, 255])
        
        self.min_radius = 3
        self.max_radius = 25
        self.min_area = 20
        self.min_brightness = 220
        self.detection_count = 0
        self.min_consecutive = 2
        
        # 圆形度阈值（完美圆形为1，设置较低值以允许轻微变形）
        self.circularity_threshold = 0.7
        
        # 定时发送相关
        self.last_send_time = cv2.getTickCount()
        
        # 通信协议
        self.STX = 0x02
        self.ETX = 0x03
    
    def detect_laser(self, frame):
        """检测红色激光点"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, self.laser_lower_red1, self.laser_upper_red1)
        mask2 = cv2.inRange(hsv, self.laser_lower_red2, self.laser_upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                
                if self.min_radius < radius < self.max_radius:
                    # 计算圆形度: circularity = 4 * π * area / perimeter^2
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        
                        # 只有圆形度达到阈值才认为是激光点
                        if circularity >= self.circularity_threshold:
                            x_int, y_int = int(x), int(y)
                            if 0 <= y_int < self.height and 0 <= x_int < self.width:
                                brightness = frame[y_int, x_int][2]
                                if brightness >= self.min_brightness:
                                    return (int(x), int(y), int(radius))
        
        return None
    
    def calculate_offset(self, laser_x, laser_y):
        """计算激光点相对画面中心的偏移量"""
        return laser_x - self.center_x, laser_y - self.center_y
    
    def format_data(self, offset_x, offset_y):
        """格式化数据为通信协议格式"""
        x_str = f"{offset_x:+04d}"
        y_str = f"{offset_y:+04d}"
        return bytes([self.STX]) + f"X{x_str}Y{y_str}".encode() + bytes([self.ETX])
    
    def send_to_serial(self, data):
        """发送数据到串口"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(data)
                return True
            except serial.SerialException as e:
                print(f"串口发送错误: {e}")
        return False
    
    def run(self):
        """主循环"""
        print("开始激光跟踪...")
        print("按 'q' 键退出")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法读取帧")
                break
            
            laser_pos = self.detect_laser(frame)
            
            # 定时发送数据（约20ms一次，50Hz）
            current_time = cv2.getTickCount()
            time_elapsed = (current_time - self.last_send_time) / cv2.getTickFrequency()
            
            if laser_pos and time_elapsed >= 0.02:  # 20ms
                laser_x, laser_y, laser_radius = laser_pos
                self.detection_count += 1
                if self.detection_count >= self.min_consecutive:
                    offset_x, offset_y = self.calculate_offset(laser_x, laser_y)
                    data = self.format_data(offset_x, offset_y)
                    self.send_to_serial(data)
                    self.last_send_time = current_time
                    
                    # 绘制激光点圆圈
                    cv2.circle(frame, (laser_x, laser_y), laser_radius + 5, (0, 0, 255), 2)
            elif not laser_pos:
                self.detection_count = 0
            
            cv2.imshow('Laser Tracker', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        if self.ser:
            self.ser.close()
        print("程序结束")

if __name__ == '__main__':
    tracker = LaserTracker(serial_port='/dev/ttyS1', baud_rate=9600)
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("用户中断")
        tracker.cap.release()
        cv2.destroyAllWindows()
        if tracker.ser:
            tracker.ser.close()