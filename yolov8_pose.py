from ultralytics import YOLO
import cv2
import numpy as np
import sys

# 读取命令行参数
weight_path = "./runs/pose/train/weights/best.onnx"  # 模型权重文件路径
media_path = "./dataset/my_yolo/images/val/00001.jpg"  # 待检测图片路径

# 加载模型
model = YOLO(weight_path)  # 使用YOLO加载模型

# 获取类别
objs_labels = model.names  # 获取模型识别的所有类别标签
print(objs_labels)  # 输出类别标签列表

# 类别的颜色
class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # 不同类别的颜色设置

# 关键点的顺序
keypoint_list = ["1", "2"]  # 关键点对应的标签

# 关键点的颜色
keypoint_color = [(255, 0, 0), (0, 255, 0)]  # 第一个关键点为红色，第二个关键点为绿色

# 读取图片
frame = cv2.imread(media_path)  # 读取待检测图片
frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))  # 缩小图片尺寸
#
# 检测
result = list(model(frame, conf=0.3, stream=True))[0]  # 进行推理，conf=0.3表示设定的检测阈值，stream=True表示返回生成器对象
boxes = result.boxes  # 获取检测到的边界框信息
boxes = boxes.cpu().numpy()  # 将边界框信息转换为NumPy数组

# 遍历关键点
keypoints = result.keypoints  # 获取检测到的关键点信息
keypoints = keypoints.cpu().numpy()  # 转换为NumPy数组

# 绘制关键点
for keypoint in keypoints.data:
    for i in range(len(keypoint)):
        x, y, c = keypoint[i]  # 获取当前关键点的坐标和置信度
        x, y = int(x), int(y)  # 转换坐标为整数
        cv2.circle(frame, (x, y), 1, keypoint_color[i], -1)  # 绘制关键点

    if len(keypoint) >= 2:  # 如果当前有至少两个关键点
        # 绘制连接两个关键点的红色直线
        x1, y1, _ = keypoint[0]  # 获取第一个关键点坐标
        x2, y2, _ = keypoint[1]  # 获取第二个关键点坐标
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 绘制红色直线
        pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        actual_distance = pixel_distance * 0.096  # 计算实际距离
        print(f"Actual distance between keypoints: {actual_distance:.2f} mm")

        midpoqint_x = int((x1 + x2) / 2)
        midpoqint_y = int((y1 + y2) / 2)
        text_offset_y = 20
        cv2.putText(frame, f"{actual_distance:.2f} mm", (midpoqint_x, midpoqint_y + text_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# 保存处理后的图片
cv2.imwrite("result/result_onnx.jpg", frame)
print("save result.jpg")  # 输出提示信息，表示图片已保存

