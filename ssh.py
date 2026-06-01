"""
香橙派蓝牙通信服务端
用于接收手机蓝牙发送的控制指令（使用STX/ETX帧协议）
协议格式：<STX> Xxxx Yyyy <ETX>
- STX: 0x02 (帧起始符)
- Xxxx: 4位ASCII字符串，X轴偏移量（带符号，范围-320到+320）
- Yyyy: 4位ASCII字符串，Y轴偏移量（带符号，范围-240到+240）
- ETX: 0x03 (帧结束符)
"""

import bluetooth
import subprocess
import struct

# 蓝牙服务配置
SERVER_PORT = 1
BUFFER_SIZE = 1024

# 协议常量
STX = 0x02  # 帧起始符
ETX = 0x03  # 帧结束符

def setup_bluetooth():
    print("正在配置蓝牙设备...")
    try:
        # 设置蓝牙名称
        subprocess.run(["hciconfig", "hci0", "name", "OrangePi-3B"], check=True)
        # 设置蓝牙可被发现
        subprocess.run(["hciconfig", "hci0", "piscan"], check=True)
        print("蓝牙配置完成")
    except subprocess.CalledProcessError as e:
        print(f"蓝牙配置失败: {e}")

def parse_protocol_frame(data):
    """
    解析蓝牙帧协议
    格式：<STX> Xxxx Yyyy <ETX>
    返回: (x_offset, y_offset) 或 None（如果解析失败）
    """
    try:
        # 查找帧起始和结束位置
        if STX not in data or ETX not in data:
            return None
        
        start_idx = data.index(STX)
        end_idx = data.index(ETX)
        
        # 确保ETX在STX之后
        if end_idx <= start_idx:
            return None
        
        # 提取帧内容（包含STX和ETX）
        frame = data[start_idx:end_idx+1]
        
        # 转换为字符串进行解析
        frame_str = frame.decode('utf-8', errors='ignore')
        
        # 移除STX和ETX
        content = frame_str.strip(chr(STX) + chr(ETX))
        
        # 分割X和Y值
        parts = content.split()
        if len(parts) != 2:
            return None
        
        x_str = parts[0].strip()
        y_str = parts[1].strip()
        
        # 解析偏移量
        x_offset = int(x_str)
        y_offset = int(y_str)
        
        # 验证范围
        if not (-320 <= x_offset <= 320):
            print(f"X轴偏移量超出范围: {x_offset}")
            return None
        if not (-240 <= y_offset <= 240):
            print(f"Y轴偏移量超出范围: {y_offset}")
            return None
        
        return (x_offset, y_offset)
        
    except (ValueError, IndexError) as e:
        print(f"帧解析错误: {e}")
        return None

def handle_command(x_offset, y_offset):
    """处理接收到的控制指令"""
    print(f"收到指令 - X轴偏移量: {x_offset}, Y轴偏移量: {y_offset}")
    
    # 计算方向
    if x_offset > 50:
        direction_x = "右移"
    elif x_offset < -50:
        direction_x = "左移"
    else:
        direction_x = "居中"
    
    if y_offset > 50:
        direction_y = "下移"
    elif y_offset < -50:
        direction_y = "上移"
    else:
        direction_y = "居中"
    
    print(f"方向: {direction_x}, {direction_y}")
    
    # 这里可以根据需要添加具体的控制逻辑
    # 例如：控制云台电机调整角度
    # x_offset > 0 表示激光点在中心右侧，需要云台向左移动
    # y_offset > 0 表示激光点在中心下方，需要云台向上移动

def main():
    # 配置蓝牙
    setup_bluetooth()
    
    # 创建蓝牙socket
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    
    try:
        # 绑定端口
        server_sock.bind(("", SERVER_PORT))
        # 监听连接，最多允许1个连接
        server_sock.listen(1)
        
        print("等待手机蓝牙连接...")
        print(f"蓝牙服务端口: {SERVER_PORT}")
        print("请在手机蓝牙设置中搜索 'OrangePi-3B' 并配对连接")
        print("\n协议格式: <STX> Xxxx Yyyy <ETX>")
        print("  Xxxx: X轴偏移量 (-320 到 +320)")
        print("  Yyyy: Y轴偏移量 (-240 到 +240)")
        
        client_sock, client_info = server_sock.accept()
        print(f"已连接: {client_info}")
        
        while True:
            try:
                # 接收数据
                data = client_sock.recv(BUFFER_SIZE)
                if not data:
                    print("连接已断开")
                    break
                
                # 解析协议帧
                result = parse_protocol_frame(data)
                
                if result is not None:
                    x_offset, y_offset = result
                    handle_command(x_offset, y_offset)
                    
                    # 发送响应
                    response = f"ACK: X={x_offset}, Y={y_offset}\n"
                    client_sock.send(response.encode('utf-8'))
                else:
                    print(f"无效数据帧: {data}")
                
            except bluetooth.BluetoothError as e:
                print(f"蓝牙通信错误: {e}")
                break
            except KeyboardInterrupt:
                print("用户中断")
                break
                
    except Exception as e:
        print(f"服务器错误: {e}")
    finally:
        # 关闭连接
        if 'client_sock' in locals():
            client_sock.close()
        server_sock.close()
        print("蓝牙服务已关闭")

if __name__ == "__main__":
    main()
