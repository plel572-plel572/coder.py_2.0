# youtube_storage_fast.py
import cv2
import numpy as np
import os
import math
import subprocess
import sys
import re


class YouTubeEncoder:
    def __init__(self, key=None):
        self.width, self.height, self.fps = 1920, 1080, 6
        self.block_width, self.block_height, self.spacing = 24, 16, 4
        self.marker_size = 80

        self.key = key
        self.use_encryption = key is not None

        # 16 цветов: индекс == 4-битное значение
        self.colors = {
            '0000': (255, 0, 0),   '0001': (0, 255, 0),   '0010': (0, 0, 255),
            '0011': (255, 255, 0), '0100': (255, 0, 255), '0101': (0, 255, 255),
            '0110': (255, 128, 0), '0111': (128, 0, 255), '1000': (0, 128, 128),
            '1001': (128, 128, 0), '1010': (128, 0, 128), '1011': (0, 128, 0),
            '1100': (128, 0, 0),   '1101': (0, 0, 128),   '1110': (192, 192, 192),
            '1111': (255, 255, 255),
        }
        # color_array[i] соответствует 4-битному i
        self.color_array = np.array(
            [self.colors['{:04b}'.format(i)] for i in range(16)], dtype=np.uint8
        )

        # Сетка
        self.blocks_x = (self.width  - 2 * self.marker_size) // (self.block_width  + self.spacing)
        self.blocks_y = (self.height - 2 * self.marker_size) // (self.block_height + self.spacing)
        self.blocks_per_region = self.blocks_x * self.blocks_y

        # Прямоугольники блоков (y1, y2, x1, x2)
        self.block_rects = []
        for idx in range(self.blocks_per_region):
            y = idx // self.blocks_x
            x = idx %  self.blocks_x
            x1 = self.marker_size + x * (self.block_width + self.spacing)
            y1 = self.marker_size + y * (self.block_height + self.spacing)
            self.block_rects.append((y1, y1 + self.block_height, x1, x1 + self.block_width))

        # Предвычисляем карту пиксель→индекс блока (для векторной отрисовки)
        self._precompute_pixel_map()

        # Базовый кадр с маркерами (копируем каждый кадр)
        self.base_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_markers_on(self.base_frame)

        self.eof_bytes = ("█" * 64).encode('utf-8')

        print("=" * 60)
        print("🎬 YouTube ENCODER (fast)")
        print("=" * 60)
        print(f"📊 Сетка: {self.blocks_x} x {self.blocks_y} = {self.blocks_per_region} блоков/кадр")
        print(f"🎞️  FPS: {self.fps}   🔐 Шифрование: {'ВКЛ' if self.use_encryption else 'ВЫКЛ'}")

    # ---------- предвычисления ----------
    def _precompute_pixel_map(self):
        """pixel_block_idx[p] = индекс блока для пикселя p, иначе -1."""
        H, W = self.height, self.width
        pixel_idx = np.full(H * W, -1, dtype=np.int32)
        for idx, (y1, y2, x1, x2) in enumerate(self.block_rects):
            for yy in range(y1, y2):
                base = yy * W
                pixel_idx[base + x1: base + x2] = idx
        self.pixel_valid_mask = pixel_idx >= 0
        self.pixel_valid_blocks = pixel_idx[self.pixel_valid_mask]

    def _draw_markers_on(self, frame):
        ms, w, h = self.marker_size, self.width, self.height
        corners = [((0, 0), (ms, ms)), ((w - ms, 0), (w, ms)),
                   ((0, h - ms), (ms, h)), ((w - ms, h - ms), (w, h))]
        for p1, p2 in corners:
            cv2.rectangle(frame, p1, p2, (255, 255, 255), -1)
            cv2.rectangle(frame, p1, p2, (0, 0, 0), 2)
        return frame

    # ---------- утилиты ----------
    @staticmethod
    def _xor_bytes(data, key):
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
        kb = key.encode()
        reps = len(arr) // len(kb) + 1
        karr = np.frombuffer(kb * reps, dtype=np.uint8)[:len(arr)]
        return (arr ^ karr).tobytes()

    def _encrypt_data(self, data):
        return self._xor_bytes(data, self.key) if self.use_encryption else data

    @staticmethod
    def _bytes_to_indices(data):
        """Байты → массив 4-битных значений (по 2 на байт)."""
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
        out = np.empty(len(arr) * 2, dtype=np.uint8)
        out[0::2] = arr >> 4
        out[1::2] = arr & 0x0F
        return out

    def _generate_frame(self, block_colors):
        """Векторная отрисовка кадра по индексам цветов (длина ≤ blocks_per_region)."""
        frame = self.base_frame.copy()
        n = len(block_colors)
        if n >= self.blocks_per_region:
            colors_per_pixel = self.color_array[block_colors[self.pixel_valid_blocks]]
        else:
            # последний кадр: часть блоков не задана → оставляем фон
            valid = self.pixel_valid_blocks < n
            flat = frame.reshape(-1, 3)
            if valid.any():
                sel_pixels = np.where(self.pixel_valid_mask)[0][valid]
                sel_blocks = self.pixel_valid_blocks[valid]
                flat[sel_pixels] = self.color_array[block_colors[sel_blocks]]
            return frame
        frame.reshape(-1, 3)[self.pixel_valid_mask] = colors_per_pixel
        return frame

    # ---------- кодирование ----------
    def encode(self, input_file, output_file):
        print("\n📤 КОДИРОВАНИЕ\n" + "-" * 40)
        with open(input_file, 'rb') as f:
            data = f.read()
        print(f"📄 {input_file}  ({len(data)} байт)")

        payload = self._encrypt_data(data) if self.use_encryption else data
        if self.use_encryption:
            print("🔐 Данные зашифрованы")

        header = f"FILE:{os.path.basename(input_file)}:SIZE:{len(data)}|"
        print(f"📋 Заголовок: {header}")

        all_indices = np.concatenate([
            self._bytes_to_indices(header.encode('latin-1')),
            self._bytes_to_indices(payload),
            self._bytes_to_indices(self.eof_bytes),
        ])
        print(f"🎨 Всего блоков: {len(all_indices)}")

        frames_needed = math.ceil(len(all_indices) / self.blocks_per_region) + 5
        print(f"🎬 Кадров: {frames_needed}  (⏱️  {frames_needed / self.fps:.1f} сек)")

        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
            ok = self._encode_pipe(all_indices, frames_needed, output_file)
        except Exception as e:
            print(f"⚠️ ffmpeg недоступен ({e}), использую OpenCV…")
            ok = self._encode_opencv(all_indices, frames_needed, output_file)
        return ok

    def _encode_pipe(self, all_indices, frames_needed, output_file):
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}', '-pix_fmt', 'bgr24',
            '-r', str(self.fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart',
            output_file,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        n = self.blocks_per_region
        n_data = frames_needed - 5
        try:
            for i in range(n_data):
                s, e = i * n, min((i + 1) * n, len(all_indices))
                chunk = all_indices[s:e]
                proc.stdin.write(self._generate_frame(chunk).tobytes())
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  🖼️  Кадр {i + 1}/{frames_needed}")
            # защитные кадры (все блоки — индекс 0)
            pad = self._generate_frame(np.zeros(n, dtype=np.uint8))
            for _ in range(5):
                proc.stdin.write(pad.tobytes())
        finally:
            proc.stdin.close()
            proc.wait()
        return self._finalize(output_file, frames_needed)

    def _encode_opencv(self, all_indices, frames_needed, output_file):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, self.fps, (self.width, self.height))
        n = self.blocks_per_region
        n_data = frames_needed - 5
        for i in range(n_data):
            s, e = i * n, min((i + 1) * n, len(all_indices))
            out.write(self._generate_frame(all_indices[s:e]))
        pad = self._generate_frame(np.zeros(n, dtype=np.uint8))
        for _ in range(5):
            out.write(pad)
        out.release()
        return self._finalize(output_file, frames_needed)

    @staticmethod
    def _finalize(output_file, frames_needed):
        if not os.path.exists(output_file):
            return False
        size = os.path.getsize(output_file)
        print(f"\n✅ Видео: {output_file}")
        print(f"📊 Размер: {size} байт ({size / 1024 / 1024:.2f} MB)")
        print(f"🎬 Кадров: {frames_needed}   ⏱️  {frames_needed / 6:.1f} сек")
        return True


class YouTubeDecoder:
    def __init__(self, key=None):
        self.width, self.height = 1920, 1080
        self.block_width, self.block_height, self.spacing = 24, 16, 4
        self.marker_size = 80
        self.key = key

        self.colors = {
            '0000': (255, 0, 0),   '0001': (0, 255, 0),   '0010': (0, 0, 255),
            '0011': (255, 255, 0), '0100': (255, 0, 255), '0101': (0, 255, 255),
            '0110': (255, 128, 0), '0111': (128, 0, 255), '1000': (0, 128, 128),
            '1001': (128, 128, 0), '1010': (128, 0, 128), '1011': (0, 128, 0),
            '1100': (128, 0, 0),   '1101': (0, 0, 128),   '1110': (192, 192, 192),
            '1111': (255, 255, 255),
        }
        self.color_array = np.array(
            [self.colors['{:04b}'.format(i)] for i in range(16)], dtype=np.int16
        )

        self.blocks_x = (self.width  - 2 * self.marker_size) // (self.block_width  + self.spacing)
        self.blocks_y = (self.height - 2 * self.marker_size) // (self.block_height + self.spacing)
        self.blocks_per_region = self.blocks_x * self.blocks_y

        # Предвычисляем координаты центров блоков
        rows, cols = [], []
        for idx in range(self.blocks_per_region):
            y = idx // self.blocks_x
            x = idx %  self.blocks_x
            cx = self.marker_size + x * (self.block_width + self.spacing) + self.block_width  // 2
            cy = self.marker_size + y * (self.block_height + self.spacing) + self.block_height // 2
            rows.append(cy); cols.append(cx)
        self.block_rows = np.array(rows, dtype=np.int32)
        self.block_cols = np.array(cols, dtype=np.int32)

        print("=" * 60)
        print("🎬 YouTube DECODER (fast)")
        print(f"📊 Сетка: {self.blocks_x} x {self.blocks_y}   🔐 Ключ: {'ЕСТЬ' if key else 'НЕТ'}")
        print("=" * 60)

    @staticmethod
    def _xor_bytes(data, key):
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
        kb = key.encode()
        reps = len(arr) // len(kb) + 1
        karr = np.frombuffer(kb * reps, dtype=np.uint8)[:len(arr)]
        return (arr ^ karr).tobytes()

    def _decrypt_data(self, data):
        return self._xor_bytes(data, self.key) if self.key else data

    def _decode_frame(self, frame, rows, cols):
        """Возвращает массив индексов 4-битных значений длиной blocks_per_region."""
        pixels = frame[rows, cols].astype(np.int16)            # (N, 3)
        diff = pixels[:, None, :] - self.color_array[None, :, :]
        dists = np.einsum('nij,nij->ni', diff, diff)           # (N, 16)
        return np.argmin(dists, axis=1).astype(np.uint8)

    def decode(self, video_file, output_dir='.'):
        print("\n📥 ДЕКОДИРОВАНИЕ\n" + "-" * 40)
        if not os.path.exists(video_file):
            print(f"❌ Файл не найден: {video_file}"); return False

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("❌ Не открыть видео"); return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or self.width
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        print(f"📹 Кадров: {total_frames}  Разрешение: {vid_w}x{vid_h}")

        # Масштабируем координаты выборки один раз — без per-frame resize
        if (vid_w, vid_h) != (self.width, self.height):
            rows = np.round(self.block_rows * vid_h / self.height).astype(np.int32)
            cols = np.round(self.block_cols * vid_w / self.width).astype(np.int32)
        else:
            rows, cols = self.block_rows, self.block_cols

        chunks = []
        processed = 0
        t0 = cv2.getTickCount()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            chunks.append(self._decode_frame(frame, rows, cols))
            if processed % 100 == 0:
                dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
                print(f"  {processed}/{total_frames}  ({processed/dt:.1f} fps)")
        cap.release()

        dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        print(f"\n📊 Обработано {processed} кадров за {dt:.2f} с ({processed/max(dt,1e-9):.1f} fps)")

        all_idx = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)
        if len(all_idx) & 1:
            all_idx = np.append(all_idx, 0)
        bytes_data = ((all_idx[0::2] << 4) | all_idx[1::2]).astype(np.uint8).tobytes()
        print(f"📦 Получено байт: {len(bytes_data)}")

        eof = b'\xe2\x96\x88' * 64
        eof_pos = bytes_data.find(eof)
        if eof_pos > 0:
            bytes_data = bytes_data[:eof_pos]
            print(f"✅ Маркер конца на позиции {eof_pos}")

        # Заголовок
        head = bytes_data[:1000].decode('latin-1', errors='ignore')
        m = re.search(r'FILE:([^:]+):SIZE:(\d+)\|', head)
        if not m:
            print("❌ Заголовок не найден")
            out = os.path.join(output_dir, "decoded_data.bin")
            with open(out, 'wb') as f:
                f.write(bytes_data)
            print(f"💾 Данные: {out}")
            return False

        filename, filesize = m.group(1), int(m.group(2))
        print(f"✅ Заголовок: {filename}, {filesize} байт")
        hb = m.group(0).encode('latin-1')
        hp = bytes_data.find(hb)
        payload = bytes_data[hp + len(hb): hp + len(hb) + filesize]
        file_data = self._decrypt_data(payload)
        if self.key:
            print("🔓 Данные расшифрованы")

        out_path = os.path.join(output_dir, filename)
        base, ext = os.path.splitext(filename)
        i = 1
        while os.path.exists(out_path):
            out_path = os.path.join(output_dir, f"{base}_{i}{ext}"); i += 1
        with open(out_path, 'wb') as f:
            f.write(file_data)
        print(f"\n✅ Файл: {out_path}")
        print(f"📏 Размер: {len(file_data)} байт "
              f"({'✅ совпадает' if len(file_data) == filesize else '⚠️ НЕ совпадает'})")
        return len(file_data) == filesize


def read_key_from_file(key_file='key.txt'):
    try:
        if os.path.exists(key_file):
            with open(key_file, 'r', encoding='utf-8') as f:
                k = f.read().strip()
            if k:
                print(f"🔑 Ключ из {key_file}")
                return k
    except Exception as e:
        print(f"⚠️ Ошибка чтения ключа: {e}")
    return None


def main():
    if len(sys.argv) < 2:
        print("\n🎥 YouTube File Storage (fast)\n"
              "  encode <файл> [out.mp4]\n"
              "  decode <видео> [папка]\n"
              "Шифрование: файл key.txt рядом с программой")
        return

    key = read_key_from_file()

    if sys.argv[1] == "encode":
        if len(sys.argv) < 3:
            print("❌ Укажите файл"); return
        YouTubeEncoder(key).encode(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "output.mp4")
    elif sys.argv[1] == "decode":
        if len(sys.argv) < 3:
            print("❌ Укажите видео"); return
        YouTubeDecoder(key).decode(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ".")
    else:
        print(f"❌ Неизвестная команда: {sys.argv[1]}")


if __name__ == "__main__":
    main()
