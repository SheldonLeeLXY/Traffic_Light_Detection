import cv2
import numpy as np


# 从文件中读取类别名称
def load_classes(names_file):
    with open(names_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


# 加载类别名称
classes = load_classes('./weights1/voc-bosch.names')

# 使用 cv2.dnn.readNet 加载模型和权重
net = cv2.dnn.readNet('./weights1/yolov3-tiny-bosch_40000.weights', './weights1/yolov3-tiny-bosch.cfg')

# # 加载类别名称
# classes = load_classes('./weights2/obj.names')
#
# # 使用 cv2.dnn.readNet 加载模型和权重
# net = cv2.dnn.readNet('./weights2/yolov3-tiny-modified_4000.weights', './weights2/yolov3-tiny-modified.cfg')


# 设置首选的推理后端（可选，依赖于系统）
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # 可以改为 DNN_TARGET_CUDA 来使用 GPU（如果支持）

# 打开 USB 摄像头（一般为 0）
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧")
        break

    # 处理帧
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    # 处理检测结果
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 根据需要调整阈值
                # 获取边界框坐标
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # 计算框的左上角
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # 绘制边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[class_id]}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

    # 显示结果
    cv2.imshow('Real-time Traffic Light Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
