import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse

# 配置信息
CLASS_NAMES = {0: 'JZ', 1: 'DL'}
CLASS_COLORS = {
    0: (255, 255, 0),  # 青黄色 (BGR)
    1: (0, 0, 255)  # 红色 (BGR)
}


def run_yolo11_segmentation(model_path, image_path, conf_threshold=0.5, iou_threshold=0.45,
                            use_gpu=False, visualize=True, save_path=None):
    """
    YOLO11实例分割推理函数

    Args:
        model_path (str): ONNX模型路径
        image_path (str): 输入图片路径
        conf_threshold (float): 置信度阈值
        iou_threshold (float): NMS的IoU阈值
        use_gpu (bool): 是否使用GPU加速
        visualize (bool): 是否可视化结果
        save_path (str): 保存结果的路径，如果为None则不保存

    Returns:
        tuple: (boxes, segments, masks, result_image)
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    # 初始化ONNX推理会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    # 获取模型输入信息
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    model_shape = model_input.shape
    input_height, input_width = model_shape[2], model_shape[3]
    dtype = np.float16 if model_input.type == 'tensor(float16)' else np.float32

    print(f"✅ 模型加载成功: {model_path}")
    print(f"🚀 推理后端: {session.get_providers()[0]}")

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 预处理
    def preprocess(img):
        h, w = img.shape[:2]
        r = min(input_height / h, input_width / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))

        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR) if (w, h) != new_unpad else img

        dw, dh = (input_width - new_unpad[0]) / 2, (input_height - new_unpad[1]) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(114, 114, 114))

        img_in = img_padded.transpose(2, 0, 1)[::-1]
        img_in = np.ascontiguousarray(img_in, dtype=dtype) / 255.0
        return img_in[None], r, (left, top)

    # 后处理
    def postprocess(preds, ori_shape, ratio, pad):
        p = np.squeeze(preds[0]).T
        proto = np.squeeze(preds[1])

        scores = np.max(p[:, 4:-32], axis=1)
        mask = scores > conf_threshold
        p = p[mask]
        scores = scores[mask]

        if len(p) == 0:
            return [], [], []

        class_ids = np.argmax(p[:, 4:-32], axis=1)
        boxes = p[:, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2  # 转换为中心点坐标
        boxes[:, 1] -= boxes[:, 3] / 2

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) == 0:
            return [], [], []

        indices = indices.flatten()
        p = p[indices]
        class_ids = class_ids[indices]
        scores = scores[indices]

        # 将中心点坐标转为角点坐标 (xyxy)
        final_boxes = p[:, :4].copy()
        final_boxes[:, 0] -= final_boxes[:, 2] / 2  # x1
        final_boxes[:, 1] -= final_boxes[:, 3] / 2  # y1
        final_boxes[:, 2] += final_boxes[:, 0]  # x2
        final_boxes[:, 3] += final_boxes[:, 1]  # y2

        # 逆缩放和逆填充处理
        final_boxes[:, [0, 2]] -= pad[0]  # x1, x2 减去左填充
        final_boxes[:, [1, 3]] -= pad[1]  # y1, y2 减去上填充
        final_boxes /= ratio  # 逆缩放
        final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]].clip(0, ori_shape[1])  # 限制在图像范围内
        final_boxes[:, [1, 3]] = final_boxes[:, [1, 3]].clip(0, ori_shape[0])

        # 处理掩码
        mask_coeffs = p[:, -32:]
        n, mh, mw = len(p), proto.shape[1], proto.shape[2]
        masks = (mask_coeffs @ proto.reshape(32, -1)).reshape(n, mh, mw)
        masks = 1 / (1 + np.exp(-masks))

        final_masks = []
        segments = []
        for i in range(n):
            # 将掩码从模型输出尺寸(如160x160)缩放到输入尺寸(如640x640)
            m = cv2.resize(masks[i], (input_width, input_height), interpolation=cv2.INTER_LINEAR)

            # 去除letterbox填充 - 使用与边界框相同的填充参数
            m = m[int(pad[1]):int(input_height - pad[1]), int(pad[0]):int(input_width - pad[0])]

            # 缩放到原始图像尺寸
            m = cv2.resize(m, (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

            # 二值化
            m = (m > 0.5).astype(np.uint8)

            # 提取轮廓作为分割区域
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=len).reshape(-1, 2)
                segments.append(c)
            else:
                segments.append(np.zeros((0, 2)))
            final_masks.append(m)

        det_results = np.concatenate([final_boxes, scores[:, None], class_ids[:, None]], axis=1)
        return det_results, segments, np.array(final_masks)

    # 可视化
    def draw_results(img, boxes, segments, alpha=0.4):
        visual_img = img.copy()
        mask_layer = np.zeros_like(img)

        for box, seg in zip(boxes, segments):
            cls_id = int(box[5])
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))

            if seg.size > 0:
                cv2.fillPoly(mask_layer, [seg.astype(np.int32)], color)

            x1, y1, x2, y2 = box[:4].astype(np.int32)
            cv2.rectangle(visual_img, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} {box[4]:.2f}"
            cv2.putText(visual_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        res = cv2.addWeighted(mask_layer, alpha, visual_img, 1.0, 0)
        return res

    # 执行推理
    blob, ratio, pad = preprocess(image)
    preds = session.run(None, {input_name: blob})
    boxes, segments, masks = postprocess(preds, image.shape, ratio, pad)

    result_image = None
    if len(boxes) > 0:
        if visualize:
            result_image = draw_results(image, boxes, segments)
            cv2.imshow("YOLO11 Segmentation", result_image)
            print(f"💡 检测到 {len(boxes)} 个目标")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path:
            if result_image is None:
                result_image = draw_results(image, boxes, segments)
            cv2.imwrite(save_path, result_image)
            print(f"💾 结果已保存到: {save_path}")
    else:
        print("⚠️ 未检测到目标")

    return boxes, segments, masks, result_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=r'E:\Desktop\ultralytics\runs\segment\ZY_BJ_JZ\weights\best.onnx',
                        help="ONNX模型路径")
    parser.add_argument("--source", type=str, default=r"E:\Desktop\LR2HR\enhanced\enhanced_0000.tif",
                        help="输入图片路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS阈值")
    parser.add_argument("--no-gpu", action="store_true", help="不使用GPU")
    parser.add_argument("--output", type=str, help="输出文件路径")
    args = parser.parse_args()

    run_yolo11_segmentation(
        model_path=args.model,
        image_path=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        use_gpu=not args.no_gpu,
        save_path=args.output
    )
