import cv2
import numpy as np
import onnxruntime as ort


class YOLOv8Pose:
    def __init__(self, model_path, conf_thres=0.1, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 初始化ONNX Runtime
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]  # (h, w)

    def preprocess(self, img):
        # Letterbox处理（保持宽高比）
        self.orig_h, self.orig_w = img.shape[:2]
        scale = min(self.input_shape[0] / self.orig_h, self.input_shape[1] / self.orig_w)

        # 计算新尺寸和填充
        self.new_unpad = (int(self.orig_w * scale), int(self.orig_h * scale))
        self.dw = (self.input_shape[1] - self.new_unpad[0]) / 2  # 水平填充
        self.dh = (self.input_shape[0] - self.new_unpad[1]) / 2  # 垂直填充

        # 执行缩放和填充
        if (self.new_unpad[0], self.new_unpad[1]) != (self.orig_w, self.orig_h):
            img = cv2.resize(img, self.new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转换颜色通道和维度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def postprocess(self, outputs):
        # 输出形状转换 [1, 11, 8400] -> [8400, 11] 
        predictions = outputs[0][0].T

        # 过滤低置信度
        conf_mask = predictions[:, 4] > self.conf_thres
        predictions = predictions[conf_mask]
        if predictions.shape[0] == 0:
            return [], [], []

        # 转换边界框坐标 (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes = predictions[:, :4].copy()
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2)  # x1
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2)  # y1
        boxes[:, 2] += boxes[:, 0]  # x2
        boxes[:, 3] += boxes[:, 1]  # y2

        # 关键点处理 (每个目标有两个关键点，每个点含x,y,score)
        keypoints = predictions[:, 5:].reshape(-1, 2, 3)  # [N, 2, 3]

        # 坐标转换到原始图像空间
        scale = min(self.input_shape[0] / self.orig_h, self.input_shape[1] / self.orig_w)

        # 调整边界框
        boxes[:, [0, 2]] -= self.dw  # 减去水平填充
        boxes[:, [1, 3]] -= self.dh  # 减去垂直填充
        boxes /= scale
        boxes = boxes.round().astype(int)

        # 调整关键点
        keypoints[:, :, 0] -= self.dw
        keypoints[:, :, 1] -= self.dh
        keypoints[:, :, :2] /= scale
        keypoints = keypoints.round().astype(int)

        # 应用NMS
        scores = predictions[:, 4]
        indices = self.nms(boxes, scores)
        return boxes[indices], scores[indices], keypoints[indices]

    def nms(self, boxes, scores):
        # OpenCV实现的高效NMS
        return cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_thres,
            self.iou_thres
        )

    def visualize(self, image, boxes, keypoints):
        # 绘制边界框
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制关键点及连线
        for kpts in keypoints:
            # 绘制关键点
            for i, (x, y, score) in enumerate(kpts):
                if score > 0.5:
                    color = (0, 0, 255) if i == 0 else (255, 0, 0)
                    cv2.circle(image, (x, y), 5, color, -1)

            # 绘制两个关键点之间的连线
            if len(kpts) == 2 and all(kpts[:, 2] > 0.5):
                x1, y1 = kpts[0][:2]
                x2, y2 = kpts[1][:2]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        return image



if __name__ == "__main__":
    model_path = "./runs/pose/train16/weights/best.onnx"
    image_path = "./input/test.png"

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to read image from {image_path}")

    # 创建YOLOv8Pose实例
    model = YOLOv8Pose(model_path)

    # 预处理
    input_tensor = model.preprocess(img)

    # 推理
    outputs = model.session.run([model.output_name], {model.input_name: input_tensor})

    # 后处理
    boxes, scores, keypoints = model.postprocess(outputs)

    # 可视化
    result = model.visualize(img.copy(), boxes, keypoints)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
