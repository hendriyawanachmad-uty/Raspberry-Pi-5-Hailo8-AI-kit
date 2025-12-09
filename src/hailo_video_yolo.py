import sys
import time
import cv2
import numpy as np
import hailo_platform as hpf

# ==============================
# Konfigurasi umum
# ==============================
VIDEO_PATH = "Road_Traffic.mp4"   # nama file video
CAM_FPS = 30                # hanya untuk estimasi fps kalau metadata video kosong

# Threshold YOLO
CONF_THRESH = 0.4           # cukup rendah untuk memastikan deteksi muncul
NMS_IOU_THRESH = 0.5

# ==============================
# Kelas YOLO (otomatis)
# ==============================

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

CLASS_NAMES = None  # akan di-set otomatis di main()


# ==============================
# Fungsi video
# ==============================
def init_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka file video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or CAM_FPS

    print(f"[INFO] Video siap: {path}")
    print(f"[INFO] Resolusi video: {width}x{height}, FPS ~ {fps:.1f}")
    return cap


# ==============================
# Preprocess & Postprocess
# ==============================
def preprocess_frame(frame, input_shape):
    """
    Preprocess untuk Hailo YOLOv8 NMS:
      - resize ke (W, H)
      - BGR -> RGB
      - dtype uint8, TANPA /255
    """
    in_h, in_w, in_c = input_shape

    # resize ke resolusi model
    img = cv2.resize(frame, (in_w, in_h))

    # BGR (OpenCV) -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pastikan uint8, range 0..255
    img = img.astype(np.uint8)

    # (1, H, W, C)
    img = np.expand_dims(img, axis=0)
    return img


def nms(bboxes, iou_threshold=0.5):
    """
    Non-Max Suppression sederhana.
    bboxes: list [x1, y1, x2, y2, score, class_id]
    """
    if len(bboxes) == 0:
        return []

    boxes = np.array(bboxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    indices = scores.argsort()[::-1]
    keep = []

    while len(indices) > 0:
        i = indices[0]
        keep.append(bboxes[i])

        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_others = (x2[indices[1:]] - x1[indices[1:]]) * (
            y2[indices[1:]] - y1[indices[1:]]
        )
        union = area_i + area_others - inter
        iou = inter / (union + 1e-6)

        remaining = np.where(iou < iou_threshold)[0]
        indices = indices[remaining + 1]

    return keep


def postprocess_detections_yolo_nms(output_raw, frame_shape):
    """
    Decoder khusus untuk output YOLOv8 NMS Hailo (sesuai contoh resmi Python):

    Struktur tipikal:
      - output_raw: array / list shape (1, num_classes)
      - output_raw[0][cls_id] -> array shape (N, 5) untuk kelas tsb
        kolom: [ymin, xmin, ymax, xmax, score]
    """
    H_frame, W_frame = frame_shape
    bboxes = []

    # Buka dimensi batch kalau ada
    # Contoh dari forum Hailo: infer_results[key][0] → iter per-class
    arr = output_raw
    if isinstance(arr, (list, tuple)):
        # bisa jadi [batch_dim][classes] atau langsung [classes]
        if len(arr) == 0:
            return []
        # kalau arr[0] juga sequence → anggap arr[0] adalah per-class
        if isinstance(arr[0], (list, tuple, np.ndarray)):
            per_class_seq = arr[0]
        else:
            # fallback, anggap arr sendiri sudah per-class
            per_class_seq = arr
    else:
        # np.ndarray, bisa dtype object dengan shape (1, 80)
        np_arr = np.array(arr, dtype=object)
        if np_arr.ndim >= 2:
            per_class_seq = np_arr[0]
        else:
            per_class_seq = np_arr

    for cls_id, class_detections in enumerate(per_class_seq):
        dets = np.asarray(class_detections, dtype=np.float32)
        if dets.size == 0:
            continue

        # Harus (N, 5): [ymin, xmin, ymax, xmax, score]
        if dets.ndim == 1:
            # kalau (5,) jadikan (1,5)
            dets = dets.reshape(1, -1)

        if dets.shape[1] < 5:
            # format tidak sesuai
            continue

        for det in dets:
            ymin, xmin, ymax, xmax, score = det[:5]

            if score < CONF_THRESH:
                continue

            # Skala koordinat (0..1) ke pixel frame
            x1 = xmin * W_frame
            y1 = ymin * H_frame
            x2 = xmax * W_frame
            y2 = ymax * H_frame

            # Clip ke dalam frame
            x1 = float(np.clip(x1, 0, W_frame - 1))
            y1 = float(np.clip(y1, 0, H_frame - 1))
            x2 = float(np.clip(x2, 0, W_frame - 1))
            y2 = float(np.clip(y2, 0, H_frame - 1))

            bboxes.append([x1, y1, x2, y2, float(score), int(cls_id)])

    if not bboxes:
        return []

    final_boxes = nms(bboxes, iou_threshold=NMS_IOU_THRESH)
    return final_boxes


def draw_detections(frame, detections):
    """
    Gambar bounding box ke frame.
    detections: list of (x1, y1, x2, y2, score, class_id)
    """
    for (x1, y1, x2, y2, score, cls_id) in detections:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if CLASS_NAMES is not None and 0 <= cls_id < len(CLASS_NAMES):
            cls_name = CLASS_NAMES[cls_id]
        else:
            cls_name = str(cls_id)

        label = f"{cls_name}:{score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return frame


# ==============================
# HailoRT init & loop inferensi
# ==============================
def main():
    global CLASS_NAMES

    if len(sys.argv) < 2:
        print("Usage: python hailo_video_yolo_v4.py <model.hef>")
        sys.exit(1)

    hef_path = sys.argv[1]
    print(f"[INFO] Memuat HEF: {hef_path}")

    # ---------- Scan device ----------
    device_ids = hpf.Device.scan()
    if not device_ids:
        print("[ERROR] Tidak ada device Hailo yang terdeteksi.")
        sys.exit(1)

    print("[INFO] Device Hailo terdeteksi:")
    for dev_id in device_ids:
        print(f"    - {dev_id}")

    # ---------- Load HEF ----------
    hef = hpf.HEF(hef_path)

    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()
    if not input_infos or not output_infos:
        print("[ERROR] HEF tidak punya info input/output vstream yang valid.")
        sys.exit(1)

    input_info = input_infos[0]
    output_info = output_infos[0]

    print("\n[INFO] Informasi vstream:")
    print(f"  Input name   : {input_info.name}")
    print(f"  Input shape  : {input_info.shape}")
    print(f"  Output name  : {output_info.name}")
    print(f"  Output shape : {output_info.shape}")

    # ---------- Deteksi jumlah kelas dari output HEF ----------
    out_shape = output_info.shape
    print(f"[DEBUG] Shape output HEF: {out_shape}", flush=True)

    if len(out_shape) >= 1:
        num_classes = out_shape[0]
    else:
        num_classes = 80  # fallback

    print(f"[INFO] Detected num_classes dari HEF: {num_classes}", flush=True)

    if num_classes == 80:
        CLASS_NAMES = COCO_CLASS_NAMES
        print("[INFO] Menggunakan COCO class names (80 kelas).", flush=True)
    else:
        CLASS_NAMES = [f"class_{i}" for i in range(num_classes)]
        print(f"[INFO] Menggunakan generic class names: class_0..class_{num_classes-1}", flush=True)

    # Ambil shape input (tanpa batch)
    if len(input_info.shape) == 3:
        non_batch_input_shape = input_info.shape
    else:
        non_batch_input_shape = input_info.shape[1:]

    # ---------- Init video ----------
    cap = init_video(VIDEO_PATH)

    # ---------- Init Hailo VDevice & network ----------
    with hpf.VDevice() as vdev:
        configure_params = hpf.ConfigureParams.create_from_hef(
            hef,
            interface=hpf.HailoStreamInterface.PCIe
        )
        network_group = vdev.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        # Sesuai saran Hailo: input UINT8, output FLOAT32, quantized=False
        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=hpf.FormatType.UINT8,
        )
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32,
        )

        print("\n[INFO] Mengaktifkan network group & pipeline inferensi...")
        with network_group.activate(network_group_params):
            with hpf.InferVStreams(
                network_group,
                input_vstreams_params,
                output_vstreams_params,
            ) as infer_pipeline:

                print("[INFO] Mulai streaming dari video. Tekan 'q' untuk keluar.")
                prev_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("[INFO] Video selesai atau gagal membaca frame.")
                        break

                    # ---------- Preprocess ----------
                    model_input = preprocess_frame(frame, non_batch_input_shape)
                    input_data = {input_info.name: model_input}

                    # ---------- Inference ----------
                    infer_start = time.time()
                    results = infer_pipeline.infer(input_data)
                    infer_time = time.time() - infer_start
                    infer_ms = infer_time * 1000.0

                    # Ambil hasil dari vstream output sesuai nama
                    output_raw = results[output_info.name]

                    # ---------- Postprocess YOLO NMS ----------
                    detections = postprocess_detections_yolo_nms(
                        output_raw,
                        frame_shape=frame.shape[:2],
                    )
                    num_dets = len(detections)

                    # ---------- Visualisasi ----------
                    vis_frame = draw_detections(frame.copy(), detections)

                    # Info jumlah deteksi
                    info_text = f"Detections: {num_dets}"
                    color = (0, 0, 255) if num_dets == 0 else (0, 255, 0)
                    cv2.putText(
                        vis_frame,
                        info_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                    # Tampilkan beberapa deteksi (nama class + koordinat)
                    start_y = 90
                    line_height = 20
                    max_lines = 5

                    if num_dets == 0:
                        cv2.putText(
                            vis_frame,
                            "No detections above threshold",
                            (10, start_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        for i, det in enumerate(detections[:max_lines]):
                            x1, y1, x2, y2, score, cls_id = det

                            if CLASS_NAMES is not None and 0 <= cls_id < len(CLASS_NAMES):
                                cls_name = CLASS_NAMES[cls_id]
                            else:
                                cls_name = str(cls_id)

                            text = (
                                f"{cls_name}: "
                                f"({int(x1)},{int(y1)})-({int(x2)},{int(y2)}) "
                                f"s={score:.2f}"
                            )
                            y_pos = start_y + i * line_height

                            cv2.putText(
                                vis_frame,
                                text,
                                (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )

                    # FPS loop
                    now = time.time()
                    dt = now - prev_time
                    fps = 1.0 / dt if dt > 0 else 0.0
                    prev_time = now

                    cv2.putText(
                        vis_frame,
                        f"FPS: {fps:.1f} | infer: {infer_ms:.1f} ms",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    cv2.imshow("Hailo YOLO Inference - Video", vis_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[INFO] Keluar dari loop inferensi.")
                        break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
