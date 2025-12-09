# HailoSDK 4.23 + Raspberry Pi 5 + Hailo-8 AI Kit  
YOLOv8 Object Detection (Video & Webcam)

Repository ini berisi dokumentasi lengkap instalasi **HailoSDK 4.23** pada  
**Raspberry Pi 5**, termasuk:

- Instalasi driver Hailo PCIe 4.23  
- Instalasi HailoRT Runtime 4.23 (.deb)  
- Instalasi Python API Hailo (`hailo_platform`) untuk Python 3.13  
- Instalasi OpenCV + NumPy  
- Contoh kode inferensi YOLOv8 (input video & webcam)  
- Troubleshooting umum  

Repository ini **khusus** untuk **SDK versi 4.23.0**.  
Tidak kompatibel dengan versi 4.20 atau di bawahnya.

---

# 1. Perangkat Keras

- Raspberry Pi 5 (4GB / 8GB)
- Hailo-8 AI Acceleration Module (PCIe)
- Carrier board/PCIe adapter untuk Raspberry Pi 5
- MicroSD 32GB+
- Kamera:
  - USB webcam, atau
  - CSI camera yang muncul sebagai `/dev/videoX`

---

# 2. Sistem Operasi yang Didukung

- Raspberry Pi OS 64-bit (Bookworm)
- Kernel default 6.x kompatibel dengan driver PCIe Hailo

---

# 3. Mengaktifkan Virtual Environment (Opsional namun Direkomendasikan)

Disarankan untuk menjalankan seluruh proyek ini di dalam **Python virtual environment (venv)**  
agar dependensi lebih terisolasi dan tidak mengubah paket sistem.

## 3.1. Membuat virtual environment

Jalankan:

```bash
python3 -m venv venv
```
Ini akan membuat folder baru bernama `venv/` di repository Anda.

---

## 3.2. Mengaktifkan virtual environment

```bash
source venv/bin/activate
```

Jika berhasil, terminal Anda akan menampilkan awalan:

> `(venv) user@raspberrypi:~`

Artinya venv sudah aktif.

---

## 3.3. Menginstal dependensi di dalam venv

Dengan venv aktif:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Untuk menginstal Python API Hailo:

```bash
pip install hailort-4.23.0-cp313-cp313-linux_aarch64.whl
```

---

## 3.4. Menonaktifkan virtual environment

Jika ingin keluar dari venv:

```bash
deactivate
```

---

# 4. Instalasi HailoSDK 4.23 (File Asli)

File yang harus diunduh dari portal Hailo:

1. `hailort-pcie-driver_4.23.0_all.deb`  
2. `hailort_4.23.0_arm64.deb`  
3. `hailort-4.23.0-cp313-cp313-linux_aarch64.whl`  
   (untuk Python **3.13**, default Raspberry Pi OS terbaru)

---

## 4.1. Install Driver PCIe 4.23

```bash
sudo dpkg -i hailort-pcie-driver_4.23.0_all.deb
sudo modprobe hailo_pci
```

Cek status driver:
```bash
lsmod | grep hailo
```

---

## 4.2. Install HailoRT Runtime 4.23

Runtime berupa paket .deb, install dengan:

```bash
sudo dpkg -i hailort_4.23.0_arm64.deb
```

Tes perintah dasar:

```bash
hailortcli --version
```

Output normal:

> `hailortcli version 4.23.0`

---

## 4.3. Install Python API (Hailo Platform) untuk Python 3.13

```bash
pip install hailort-4.23.0-cp313-cp313-linux_aarch64.whl
```

Cek:

```bash
python3 - << 'EOF'
import hailo_platform as h
print("Hailo Platform Version:", h.__version__)
EOF
```

Harus muncul:

> `Hailo Platform Version: 4.23.0`

---

## 4.4. Verifikasi Device Hailo-8

```bash
hailortcli scan
```

Output:

> `Hailo Devices:`
> `Device: 0001:04:00.0`


Jika tidak muncul → cek PCIe connector dan driver.

---

# 5. Instalasi OpenCV + NumPy

```bash
sudo apt install -y python3-opencv libopencv-dev
python3 -m pip install numpy
```

Testing:

```bash
python3 - << 'EOF'
import cv2, numpy as np
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
EOF
```

# 6. Menjalankan YOLOv8 .hef di Hailo SDK 4.23

Download yolov8s.hef dari Hailo Model Zoo, lalu simpan di:

> `~/models/yolov8s.hef`

---

# 7. Struktur Repository
hailo8-raspberrypi5-yolov8/
- `README`.md
- `requirements`.txt
- samples/
  - `cars`.mp4
  - `bikes`.mp4
  - `motorbikes`.mp4
  - `peoples`.mp4
- models/
  - `yolov8s`.hef
- src/
  - `hailo_video_yolo`.py
  - `hailo_webcam_yolo`.py

---

# 8. Inferensi YOLOv8 dari File Video

```bash
python3 src/hailo_video_yolo.py models/yolov8s.hef
```

Hasil:

> `Bounding box`
> `Label class (person, car, dll)`
> `FPS dan inference time`

---

# 9. Inferensi YOLOv8 dari Webcam

Default kamera (/dev/video0):

```bash
python3 src/hailo_webcam_yolo.py models/yolov8s.hef
```

Jika menggunakan device lain (misalnya /dev/video8):

```bash
python3 src/hailo_webcam_yolo.py models/yolov8s.hef 8
```

# 10. Troubleshooting (HailoSDK 4.23)
###### ❌ `hailortcli` device scan kosong

Solusi:

```bash
lspci -nn | grep -i hailo
dmesg | grep -i hailo
```

Pastikan driver PCIe terinstal.

###### ❌ Python tidak menemukan `hailo_platform`
- Pastikan `Python 3.13`
- Pastikan wheel yang digunakan: `cp313-cp313-linux_aarch64`

Cek: 

```bash
pip list | grep hailo
```

###### ❌ `Detections selalu 0` (tidak ada bounding box)

Solusi:
- Preprocess harus RGB uint8, tidak dibagi 255
- Model .hef harus YOLOv8 COCO
- Turunkan threshold: `CONF_THRESH` = 0.1

###### ❌ `Error “setting array element with a sequence”`

- Output NMS pada SDK 4.23 adalah ragged array
- Gunakan decoder per-class seperti script di repo ini.

---
