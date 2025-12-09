import cv2

def video_capture(device_index=8):
    # Paksa backend V4L2
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Tidak bisa membuka kamera pada index {device_index}")
        return

    # Set resolusi (sesuaikan kebutuhan)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set format MJPG agar fps stabil dan read() tidak gagal
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Streaming dimulai...")
    print("Tekan 'c' untuk capture, 'q' untuk keluar.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        # Tampilkan frame
        cv2.imshow("USB Webcam (V4L2 + MJPG)", frame)

        key = cv2.waitKey(1) & 0xFF

        # Simpan frame ketika 'c' ditekan
        if key == ord('c'):
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Gambar disimpan sebagai {filename}")
            frame_count += 1

        # Keluar jika 'q'
        elif key == ord('q'):
            print("Keluar dari video capture.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_capture(8)   # kalau webcam ada di /dev/video1 → ganti jadi video_capture(1)
