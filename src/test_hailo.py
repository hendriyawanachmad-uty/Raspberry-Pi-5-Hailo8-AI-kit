import sys
import numpy as np
import hailo_platform as hpf


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_hef_inference.py <model.hef>")
        sys.exit(1)

    hef_path = sys.argv[1]
    print(f"[INFO] Memuat HEF: {hef_path}")

    # ---------- 1. Scan device ----------
    device_ids = hpf.Device.scan()
    if not device_ids:
        print("[ERROR] Tidak ada device Hailo yang terdeteksi.")
        sys.exit(1)

    print("[INFO] Device Hailo terdeteksi:")
    for dev_id in device_ids:
        print(f"    - {dev_id}")

    # ---------- 2. Load HEF ----------
    hef = hpf.HEF(hef_path)

    input_infos = hef.get_input_vstream_infos()
    output_infos = hef.get_output_vstream_infos()

    if not input_infos or not output_infos:
        print("[ERROR] HEF tidak punya info input/output vstream yang valid.")
        sys.exit(1)

    # Untuk contoh ini, kita pakai 1 input dan 1 output pertama
    input_info = input_infos[0]
    output_info = output_infos[0]

    print("\n[INFO] Informasi vstream:")
    print(f"  Input name   : {input_info.name}")
    print(f"  Input shape  : {input_info.shape}")
    print(f"  Output name  : {output_info.name}")
    print(f"  Output shape : {output_info.shape}")

    # ---------- 3. Buat VDevice dan konfigurasi network ----------
    # VDevice akan memilih device yang tersedia (bisa juga diisi 'device_ids=device_ids')
    with hpf.VDevice() as vdev:
        # Parameter konfigurasi dari HEF
        configure_params = hpf.ConfigureParams.create_from_hef(
            hef,
            interface=hpf.HailoStreamInterface.PCIe  # atau Ethernet kalau pakai Ethernet
        )

        # Konfigurasi HEF di device → hasilnya 1 atau lebih network_group
        network_group = vdev.configure(hef, configure_params)[0]

        # Parameter aktivasi network group
        network_group_params = network_group.create_params()

        # ---------- 4. Siapkan parameter vstream ----------
        # Di contoh ini kita pakai format float32 (quantized=False),
        # kalau model pakai INT8, bisa diganti sesuai kebutuhan.
        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32
        )
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=hpf.FormatType.FLOAT32
        )

        print("\n[INFO] Mengaktifkan network group dan membuat infer pipeline...")

        # ---------- 5. Aktivasi dan jalankan 1x inferensi ----------
        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group,
                                   input_vstreams_params,
                                   output_vstreams_params) as infer_pipeline:

                # Siapkan input dummy sesuai bentuk input vstream
                input_shape = input_info.shape

                # Jika shape = (H, W, C) → tambahkan batch dim di depan
                if len(input_shape) == 3:
                    # (H, W, C) → (1, H, W, C)
                    tensor_shape = (1, *input_shape)
                else:
                    # Kalau sudah ada batch di HEF, langsung pakai
                    tensor_shape = tuple(input_shape)

                # Random input [0..1], tipe float32
                input_tensor = np.random.rand(*tensor_shape).astype(np.float32)

                # Kemas ke dict sesuai dengan nama vstream input
                input_data = {
                    input_info.name: input_tensor
                }

                print(f"[INFO] Menjalankan satu kali inferensi dengan input shape {tensor_shape}...")

                results = infer_pipeline.infer(input_data)

                # Ambil output dengan nama vstream output
                output_tensor = results[output_info.name]

                print("[INFO] Inferensi selesai.")
                print(f"  Output dtype  : {output_tensor.dtype}")
                print(f"  Output shape  : {output_tensor.shape}")

                # Tampilkan sedikit nilai output
                flat = output_tensor.flatten()
                n_show = min(10, flat.size)
                print(f"  Sample output[0:{n_show}]: {flat[:n_show]}")


if __name__ == "__main__":
    main()
