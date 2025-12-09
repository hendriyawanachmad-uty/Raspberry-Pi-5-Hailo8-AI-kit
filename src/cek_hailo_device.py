from hailo_platform import Device

def main():
    device_ids = Device.scan()
    if not device_ids:
        print("Tidak ada device Hailo yang terdeteksi.")
        return

    print("Device Hailo terdeteksi:")
    for dev_id in device_ids:
        print(" -", dev_id)

    # Kalau mau langsung buka satu device:
    device = Device(device_id=device_ids[0])
    print("Terhubung ke device:", device.device_id)
    device.release()

if __name__ == "__main__":
    main()
