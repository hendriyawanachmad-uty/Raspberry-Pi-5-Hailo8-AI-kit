# Raspberry-Pi-5-Hailo8-AI-kit
Panduan penggunaan modul Hailo8 AI Kit pada Raspberry Pi 5 untuk deteksi atau pengenalan obyek visual.

# 1. Instalasi HailoSDK 4.23 untuk Raspberry Pi 5
Panduan ini menggunakan file **HailoSDK versi 4.23.0** yang *benar-benar diperlukan* untuk Raspberry Pi 5:

File yang harus Anda unduh dari portal Hailo:

1. `hailort-pcie-driver_4.23.0_all.deb`  
2. `hailort_4.23.0_arm64.deb`  
3. `hailort-4.23.0-cp313-cp313-linux_aarch64.whl`

Repository ini ditulis khusus untuk SDK versi *4.23.0*.

---

# 1.1. Install Driver PCIe untuk Hailo-8 (4.23.0)

Install driver:

```bash
sudo dpkg -i hailort-pcie-driver_4.23.0_all.deb
sudo modprobe hailo_pci
