import numpy as np
import librosa
import cv2
import moviepy.editor as mpy
from tqdm import tqdm
import os
import random
import time

"""
VHSスタイル音楽ビジュアライザー

このスクリプトは、指定された画像と音楽ファイルを使用して、
古いVHSテープのような見た目のビジュアライゼーションを生成します。

主な特徴:
- VHSスタイルのノイズとグリッチ
- スキャンラインとトラッキングエラー
- カラーブリーディングとカラーシフト
- タイムコード表示
- テープの劣化を模したエフェクト
"""


class VHSMusicVisualizer:
    """
    VHSスタイルの音楽ビジュアライザークラス

    音楽ファイルと画像ファイルを使用して、VHSテープのような
    ビジュアライゼーションを生成します。
    """

    def __init__(self, image_path, audio_path, output_path):
        """
        コンストラクタ

        :param image_path: str 画像ファイルのパス
        :param audio_path: str 音楽ファイルのパス
        :param output_path: str 出力ビデオファイルのパス
        """
        self.image_path = image_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.fps = 24
        self.vhs_noise_intensity = 0.4
        self.tracking_error_chance = 0.07
        self.color_bleeding_intensity = 0.6
        self.scanline_intensity = 0.3
        self.glitch_chance = 0.15
        self.timecode_enabled = True
        self.start_time = time.time()

        # VHSカラーパレット - 彩度を下げた色
        self.color_palette = [
            (180, 180, 240),  # 薄い青
            (240, 180, 180),  # 薄い赤
            (180, 240, 180),  # 薄い緑
            (220, 220, 180),  # 薄い黄色
            (220, 180, 220),  # 薄い紫
        ]

    def load_audio(self):
        """
        音楽ファイルをロードし、サンプリングレートとフレーム数を計算します。
        """
        print("Loading audio...")
        self.y, self.sr = librosa.load(self.audio_path)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.n_frames = int(self.duration * self.fps)

    def analyze_audio(self):
        """
        音楽ファイルを解析し、メルスペクトログラム、オンセット強度、ビートを計算します。
        """
        print("Analyzing audio...")
        hop_length = self.sr // self.fps
        S = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_mels=64, fmax=8000, hop_length=hop_length
        )
        self.S_dB = librosa.power_to_db(S, ref=np.max)
        self.onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=hop_length
        )
        self.beats = librosa.beat.beat_track(
            onset_envelope=self.onset_env, sr=self.sr, hop_length=hop_length
        )[1]
        self.max_onset = np.max(self.onset_env)

    def load_image(self):
        """
        画像ファイルをロードし、RGB形式に変換します。
        """
        print("Loading image...")
        self.img = cv2.imread(self.image_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 画像のアスペクト比を維持しながらリサイズ
        target_height = 960  # 9:16アスペクト比の縦長解像度
        target_width = 540

        h, w = self.img.shape[:2]
        aspect = w / h

        if aspect > target_width / target_height:  # 横長の画像
            new_width = target_width
            new_height = int(new_width / aspect)
        else:  # 縦長の画像
            new_height = target_height
            new_width = int(new_height * aspect)

        self.img = cv2.resize(self.img, (new_width, new_height))

        # 黒い背景に画像を配置
        background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        background[
            y_offset : y_offset + new_height, x_offset : x_offset + new_width
        ] = self.img

        self.img = background
        self.height, self.width, _ = self.img.shape

    def initialize_video(self):
        """
        ビデオファイルを初期化し、書き込み準備を行います。
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 9:16アスペクト比の縦長動画を作成
        self.video = cv2.VideoWriter(
            "temp_output.mp4", fourcc, self.fps, (self.width, self.height)
        )

    def apply_vhs_noise(self, frame, intensity):
        """
        VHSノイズを適用します。
        高速化版

        :param frame: np.ndarray 現在のフレーム
        :param intensity: float ノイズの強度
        :return: np.ndarray 更新されたフレーム
        """
        # ノイズ生成を高速化（サイズを小さくしてからリサイズ）
        height, width = frame.shape[:2]
        scale_factor = 0.5  # ノイズの解像度を半分に
        small_noise = np.random.normal(0, 15, (int(height * scale_factor), int(width * scale_factor), 3)).astype(np.int16)
        noise = cv2.resize(small_noise, (width, height), interpolation=cv2.INTER_LINEAR)

        # ノイズを適用
        frame_int16 = frame.astype(np.int16) + (noise * intensity).astype(np.int16)
        return np.clip(frame_int16, 0, 255).astype(np.uint8)

    def apply_scanlines(self, frame, intensity):
        """
        スキャンラインエフェクトを適用します。
        高速化版

        :param frame: np.ndarray 現在のフレーム
        :param intensity: float スキャンラインの強度
        :return: np.ndarray 更新されたフレーム
        """
        height, width = frame.shape[:2]

        # ベクトル化した高速処理
        # 奇数行にのみスキャンラインを適用
        frame_float = frame.astype(np.float32)
        frame_float[::2, :] = frame_float[::2, :] * (1.0 - intensity)

        # ムラを加えるのはランダムな一部の行のみにして高速化
        if np.random.random() < 0.3:  # 30%の確率でムラを加える
            random_rows = np.random.choice(range(0, height, 2), size=min(10, height//2), replace=False)
            for y in random_rows:
                noise = np.random.uniform(0.9, 1.0, width)
                frame_float[y, :, 0] *= noise
                frame_float[y, :, 1] *= noise
                frame_float[y, :, 2] *= noise

        return np.clip(frame_float, 0, 255).astype(np.uint8)

    def apply_color_bleeding(self, frame, intensity):
        """
        カラーブリーディングエフェクトを適用します。
        高速化版

        :param frame: np.ndarray 現在のフレーム
        :param intensity: float カラーブリーディングの強度
        :return: np.ndarray 更新されたフレーム
        """
        # チャンネルを分割
        b, g, r = cv2.split(frame)

        # シフト量を計算
        r_shift_amount = int(6 * intensity)
        b_shift_amount = int(5 * intensity)
        g_shift_amount = int(2 * intensity)

        # シフトを高速化（np.rollを使用）
        if r_shift_amount > 0:
            r_shifted = np.roll(r, r_shift_amount, axis=1)
            r_shifted[:, :r_shift_amount] = 0  # 左端をゼロ埋め
        else:
            r_shifted = r.copy()

        if b_shift_amount > 0:
            b_shifted = np.roll(b, -b_shift_amount, axis=1)
            b_shifted[:, -b_shift_amount:] = 0  # 右端をゼロ埋め
        else:
            b_shifted = b.copy()

        if g_shift_amount > 0:
            g_shifted = np.roll(g, g_shift_amount, axis=1)
            g_shifted[:, :g_shift_amount] = 0  # 左端をゼロ埋め
        else:
            g_shifted = g.copy()

        # 各チャンネルの強度を調整（ベクトル化）
        r_shifted = r_shifted * 1.2
        b_shifted = b_shifted * 0.85
        g_shifted = g_shifted * 0.95

        # 値の範囲をクリップ
        r_shifted = np.clip(r_shifted, 0, 255).astype(np.uint8)
        b_shifted = np.clip(b_shifted, 0, 255).astype(np.uint8)
        g_shifted = np.clip(g_shifted, 0, 255).astype(np.uint8)

        # チャンネルを結合
        return cv2.merge([b_shifted, g_shifted, r_shifted])

    def apply_tracking_error(self, frame, strength):
        """
        トラッキングエラーエフェクトを適用します。

        :param frame: np.ndarray 現在のフレーム
        :param strength: float エラーの強度
        :return: np.ndarray 更新されたフレーム
        """
        height, width = frame.shape[:2]

        # ランダムな垂直シフト
        shift_y = int(strength * height * 0.1)
        if shift_y > 0:
            frame = np.roll(frame, shift_y, axis=0)
            # 上部をノイズで埋める
            frame[:shift_y, :] = np.random.randint(
                0, 255, (shift_y, width, 3), dtype=np.uint8
            )

        # VHSヘッドの切り替わりを再現する水平ノイズライン
        if np.random.random() < 0.3:  # 30%の確率で発生
            # ランダムな位置に水平ノイズラインを追加
            for _ in range(np.random.randint(1, 3)):
                y_pos = np.random.randint(0, height)
                line_height = np.random.randint(2, 5)
                noise_line = np.random.randint(
                    200, 255, (line_height, width, 3), dtype=np.uint8
                )

                # ノイズラインに歧みを加える
                for x in range(width):
                    offset = int(np.sin(x * 0.1) * 2)
                    y = min(max(0, y_pos + offset), height - line_height)
                    frame[y : y + line_height, x] = noise_line[:, x]

        # 画面上部や下部にノイズバーを追加
        if np.random.random() < 0.2:  # 20%の確率で発生
            bar_height = np.random.randint(5, 15)
            if np.random.random() < 0.5:  # 上部に追加
                frame[:bar_height, :] = np.random.randint(
                    0, 255, (bar_height, width, 3), dtype=np.uint8
                )
            else:  # 下部に追加
                frame[-bar_height:, :] = np.random.randint(
                    0, 255, (bar_height, width, 3), dtype=np.uint8
                )

        return frame

    def apply_vhs_glitch(self, frame, intensity):
        """
        VHSグリッチエフェクトを適用します。
        高速化版

        :param frame: np.ndarray 現在のフレーム
        :param intensity: float グリッチの強度
        :return: np.ndarray 更新されたフレーム
        """
        # フレームレートを上げるため、確率でスキップ
        if np.random.random() > 0.3 * intensity:  # 発生確率を下げる
            return frame  # グリッチを適用しない

        height, width = frame.shape[:2]
        glitch_frame = frame.copy()

        # グリッチの種類をランダムに選択（全てを適用するのではなく、一部のみを適用）
        glitch_type = np.random.choice(['vertical', 'horizontal', 'color', 'fade'], p=[0.2, 0.4, 0.3, 0.1])

        if glitch_type == 'vertical':
            # 垂直方向のグリッチ
            vertical_shift = np.random.randint(5, 20)
            glitch_frame = np.roll(glitch_frame, vertical_shift, axis=0)

            # ノイズラインを追加（数を減らす）
            num_lines = np.random.randint(1, 4)  # ライン数を減らす
            for _ in range(num_lines):
                y = np.random.randint(0, height)
                h = np.random.randint(2, 6)  # 高さを小さく
                noise = np.random.randint(0, 255, (h, width, 3), dtype=np.uint8)
                if 0 <= y < height - h:
                    glitch_frame[y:y+h, :] = noise

        elif glitch_type == 'horizontal':
            # 水平グリッチ（数を減らす）
            num_glitches = np.random.randint(1, 3)  # グリッチ数を減らす
            for _ in range(num_glitches):
                y_start = np.random.randint(0, height - 10)
                y_end = min(y_start + np.random.randint(5, 15), height)
                x_shift = np.random.randint(-15, 15)

                if x_shift == 0:
                    continue

                # np.rollを使用して高速化
                if x_shift > 0:
                    glitch_frame[y_start:y_end] = np.roll(frame[y_start:y_end], x_shift, axis=1)
                    glitch_frame[y_start:y_end, :x_shift] = 0
                else:
                    glitch_frame[y_start:y_end] = np.roll(frame[y_start:y_end], x_shift, axis=1)
                    glitch_frame[y_start:y_end, x_shift:] = 0

        elif glitch_type == 'color':
            # カラーグリッチ
            channel = np.random.randint(0, 3)
            shift = np.random.randint(-10, 10)
            glitch_frame[:, :, channel] = np.roll(glitch_frame[:, :, channel], shift, axis=1)

        elif glitch_type == 'fade':
            # 色あせエフェクト（領域を小さく）
            fade_x = np.random.randint(0, width - 30)
            fade_y = np.random.randint(0, height - 30)
            fade_w = np.random.randint(30, min(100, width - fade_x))
            fade_h = np.random.randint(30, min(100, height - fade_y))

            # 色あせ領域を作成
            fade_region = glitch_frame[fade_y:fade_y+fade_h, fade_x:fade_x+fade_w].copy()
            fade_region = cv2.addWeighted(fade_region, 0.7, np.ones_like(fade_region) * 30, 0.3, 0)
            glitch_frame[fade_y:fade_y+fade_h, fade_x:fade_x+fade_w] = fade_region

        return glitch_frame

    def add_timecode(self, frame, frame_num):
        """
        VHSスタイルのタイムコードを追加します。

        :param frame: np.ndarray 現在のフレーム
        :param frame_num: int フレーム番号
        :return: np.ndarray 更新されたフレーム
        """
        seconds = frame_num / self.fps
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        frames = frame_num % self.fps

        timecode = f"{minutes:02d}:{seconds:02d}:{frames:02d}"

        # タイムコードを画像に追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            timecode,
            (self.width - 120, self.height - 20),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    def add_vhs_label(self, frame):
        """
        VHSラベルを追加します。

        :param frame: np.ndarray 現在のフレーム
        :return: np.ndarray 更新されたフレーム
        """
        # VHSラベルを画像に追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "VHS", (20, 30), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "SP", (20, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def apply_vhs_filter(self, frame, energy, current_onset, frame_num=0):
        """
        すべてのVHSエフェクトを適用します。
        高速化版

        :param frame: np.ndarray 現在のフレーム
        :param energy: float 音楽のエネルギー
        :param current_onset: float 現在のオンセット強度
        :return: np.ndarray 更新されたフレーム
        """
        # エフェクトの適用順序を最適化
        # 必須のエフェクトのみを常に適用し、その他は確率で適用

        # 1. 基本的なノイズとスキャンライン（常に適用）
        frame = self.apply_vhs_noise(frame, self.vhs_noise_intensity + energy * 0.15)
        frame = self.apply_scanlines(frame, self.scanline_intensity + energy * 0.05)

        # 2. カラーブリーディング（常に適用）
        frame = self.apply_color_bleeding(frame, self.color_bleeding_intensity + energy * 0.15)

        # 3. 波状の揺れエフェクト（内部で確率判定）
        frame = self.apply_vhs_wave_distortion(frame, frame_num, energy * 0.3)

        # 4. トラッキングエラーとグリッチは確率を下げて適用
        # ビートに合わせて発生確率を上げる
        if energy > 0.2 or current_onset > self.max_onset * 0.5:  # エネルギーが高い場合のみ
            if np.random.random() < self.tracking_error_chance * 0.3 + current_onset / (self.max_onset * 30):
                frame = self.apply_tracking_error(frame, 0.3 + current_onset / self.max_onset)

            if np.random.random() < self.glitch_chance * 0.3 + current_onset / (self.max_onset * 15):
                frame = self.apply_vhs_glitch(frame, 0.3 + current_onset / self.max_onset)

        # 5. ノイズバンドは内部で確率判定
        frame = self.apply_vhs_noise_bands(frame, energy, current_onset)

        # 6. 色調整（常に適用）
        frame = cv2.convertScaleAbs(frame, alpha=0.85, beta=12)

        # 7. HSV変換と彩度調整（常に適用）
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.75  # 彩度を75%に
        hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180  # 色相をシフト
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return frame

    def apply_vhs_wave_distortion(self, frame, frame_num, energy):
        """
        VHS特有の波状の揺れエフェクトを適用します。
        高速化版

        :param frame: np.ndarray 現在のフレーム
        :param frame_num: int フレーム番号
        :param energy: float 音楽のエネルギー
        :return: np.ndarray 更新されたフレーム
        """
        # フレームレートを上げるため、全てのフレームではなく一部のフレームのみに揺れを適用
        # ビートに合わせて揺れを強めるようにする
        if np.random.random() > 0.7 and energy < 0.1:  # エネルギーが低い場合は確率を下げる
            return frame  # 揺れを適用しない

        height, width = frame.shape[:2]

        # 波の振幅（エネルギーに応じて変化）
        base_amplitude = 1.5 + energy * 2  # 揺れの大きさ

        # 波の週期とジッターを計算
        period = 0.1 + np.sin(frame_num * 0.01) * 0.05
        speed_variation = np.sin(frame_num * 0.03) * 2
        jitter_x = int(np.random.normal(0, 0.5))
        jitter_y = int(np.random.normal(0, 0.2))

        # ベクトル化した高速処理のための準備
        # 各行のオフセットを一度に計算
        y_indices = np.arange(height)
        wave1 = np.sin(y_indices * period + frame_num * 0.05)
        wave2 = np.sin(y_indices * period * 0.7 + frame_num * 0.02) * 0.5
        combined_wave = wave1 + wave2 + speed_variation * 0.2
        offsets = (base_amplitude * combined_wave).astype(int) + jitter_x

        # ワープマップを作成して高速化
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            offset = offsets[y]
            for x in range(width):
                src_x = x + offset
                src_y = y + jitter_y

                # 画像の端からはみ出さないようにする
                src_x = max(0, min(src_x, width - 1))
                src_y = max(0, min(src_y, height - 1))

                map_x[y, x] = src_x
                map_y[y, x] = src_y

        # ワープ変換を適用
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # ブラーは確率を下げて高速化
        if np.random.random() < 0.1:  # 10%の確率に下げる
            blur_amount = np.random.randint(1, 3)
            result = cv2.GaussianBlur(result, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)

        return result

    def apply_vhs_noise_bands(self, frame, energy, current_onset):
        """
        VHS特有のノイズバンド（横方向のノイズの波）エフェクトを適用します。
        高速化版

        :param frame: np.ndarray 現在のフレーム
        :param energy: float 音楽のエネルギー
        :param current_onset: float 現在のオンセット強度
        :return: np.ndarray 更新されたフレーム
        """
        # フレームレートを上げるため、確率でスキップ
        if np.random.random() > 0.3 + energy * 0.3:  # エネルギーが高いときは発生確率を上げる
            return frame  # ノイズバンドを適用しない

        height, width = frame.shape[:2]
        result = frame.copy()

        # ノイズバンドの数を減らす
        num_bands = max(1, int(energy * 1.5))  # 最少で1つ、エネルギーに応じて最大1、2個

        # オンセット強度に応じてノイズバンドの強度を変化
        intensity = 0.1 + (current_onset / self.max_onset) * 0.2

        # バンドの位置を事前に計算
        y_positions = np.random.randint(0, height, num_bands)
        band_heights = np.random.randint(2, 6, num_bands)  # バンドの高さを少し下げる
        band_opacities = np.random.uniform(0.3, 0.7, num_bands) * intensity

        for i in range(num_bands):
            y_pos = y_positions[i]
            band_height = band_heights[i]
            band_opacity = band_opacities[i]

            # ノイズバンドを一度に生成
            noise = np.random.randint(180, 255, (band_height, width, 3), dtype=np.uint8)

            # 波のような効果を追加（単純化）
            # 波の計算をベクトル化
            x_indices = np.arange(width)
            wave_offsets = (np.sin(x_indices * 0.05 + np.random.rand() * 10) * 2).astype(int)

            for x in range(width):
                wave_offset = wave_offsets[x]
                y = y_pos + wave_offset

                # 画像の端からはみ出さないようにする
                if 0 <= y < height - band_height:
                    # バンド全体を一度に適用
                    result[y:y+band_height, x] = cv2.addWeighted(
                        result[y:y+band_height, x],
                        1 - band_opacity,
                        noise[:, x],
                        band_opacity,
                        0
                    )

        # 水平ノイズラインは確率を下げる
        if np.random.random() < 0.03 + energy * 0.05:  # 発生確率をさらに下げる
            line_y = np.random.randint(0, height)
            line_height = np.random.randint(1, 3)
            line_color = (255, 255, 255)
            cv2.line(result, (0, line_y), (width, line_y), line_color, line_height)

        return result

    def generate_frames(self):
        """
        音楽の特徴に基づいて各フレームを生成し、ビデオファイルに書き込みます。
        """

        for frame_num in tqdm(
            range(self.n_frames), desc="Generating frames", unit="frame"
        ):
            # 音楽データのインデックスを計算
            audio_idx = int(frame_num * len(self.y) / self.n_frames)
            chunk = self.y[audio_idx : audio_idx + self.sr // self.fps]

            # 現在のフレームの音楽特徴を取得
            spec_frame = self.S_dB[:, min(frame_num, self.S_dB.shape[1] - 1)]
            current_onset = self.onset_env[min(frame_num, len(self.onset_env) - 1)]
            energy = np.mean(np.abs(chunk)) * 10

            # ベースフレームを作成
            frame = self.img.copy()

            # ビートに合わせてエフェクトを強調
            is_beat = frame_num in self.beats
            beat_intensity = 1.5 if is_beat else 1.0

            # VHSエフェクトを適用
            frame = self.apply_vhs_filter(
                frame,
                energy * beat_intensity,
                current_onset * beat_intensity,
                frame_num,
            )

            # VHSノイズバンドエフェクトを適用
            frame = self.apply_vhs_noise_bands(
                frame, energy * beat_intensity, current_onset * beat_intensity
            )

            # タイムコードを追加
            if self.timecode_enabled:
                frame = self.add_timecode(frame, frame_num)

            # VHSラベルを追加
            frame = self.add_vhs_label(frame)

            # フレームをビデオに書き込み
            self.video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        self.video.release()

    def combine_audio_and_video(self):
        """
        音楽とビデオを結合し、最終的なビデオファイルを生成します。
        """
        print("Combining audio and video...")

        # FFmpegを直接使用して音声とビデオを結合
        try:
            import subprocess

            cmd = [
                "ffmpeg",
                "-i",
                "temp_output.mp4",  # ビデオ入力
                "-i",
                self.audio_path,  # 音声入力
                "-c:v",
                "copy",  # ビデオコーデックをコピー
                "-c:a",
                "aac",  # 音声コーデックをAACに設定
                "-shortest",  # 最短のストリームの長さに合わせる
                "-y",  # 既存ファイルを上書き
                self.output_path,  # 出力ファイル
            ]
            subprocess.run(cmd, check=True)
            print(f"Successfully combined video and audio to {self.output_path}")
        except Exception as e:
            print(f"Error combining video and audio: {e}")
            print("Falling back to MoviePy method...")

            # MoviePyを使用するフォールバック方法
            try:
                video = mpy.VideoFileClip("temp_output.mp4")
                # fpsを明示的に設定
                video.fps = 30.0  # 固定値を使用
                audio = mpy.AudioFileClip(self.audio_path)
                if video.duration > audio.duration:
                    video = video.subclip(0, audio.duration)
                else:
                    audio = audio.subclip(0, video.duration)

                final_video = video.set_audio(audio)
                final_video.write_videofile(
                    self.output_path, fps=30.0, codec="libx264", audio_codec="aac"
                )
            except Exception as e2:
                print(f"MoviePy method also failed: {e2}")
                print("Please manually combine the video and audio files.")

        # 一時ファイルを削除
        if os.path.exists("temp_output.mp4"):
            os.remove("temp_output.mp4")

    def create_visualization(self):
        """
        ビジュアライゼーションを作成するためのメインメソッド。
        """
        self.load_audio()
        self.analyze_audio()
        self.load_image()
        self.initialize_video()
        self.generate_frames()
        self.combine_audio_and_video()
        print(f"VHS style visualization created: {self.output_path}")


def get_files(extensions):
    """
    指定された拡張子を持つファイルのリストを取得します。

    :param extensions: list 拡張子のリスト
    :return: list ファイルのリスト
    """
    return [f for f in os.listdir(".") if f.lower().endswith(tuple(extensions))]


def select_file(file_type, extensions):
    """
    ユーザーにファイルを選択させます。

    :param file_type: str ファイルタイプの説明
    :param extensions: list 拡張子のリスト
    :return: str 選択されたファイルのパス
    """
    files = get_files(extensions)

    if not files:
        print(f"No {file_type} files found in the current directory.")
        return None

    print(f"Available {file_type} files:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(
                input(f"Enter the number of the {file_type} file you want to use: ")
            )
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def select_mp3_file():
    """
    ユーザーに音楽ファイルを選択させます。

    :return: str 選択された音楽ファイルのパス
    """
    return select_file("audio", [".mp3", ".wav"])


def select_image_file():
    """
    ユーザーに画像ファイルを選択させます。

    :return: str 選択された画像ファイルのパス
    """
    return select_file("image", [".png", ".jpg", ".jpeg"])


def main():
    """
    メイン関数
    """
    print("=== VHS Style Music Visualizer ===")
    print("This program creates a music visualization with retro VHS effects.")

    image_path = select_image_file()
    if not image_path:
        return

    audio_path = select_mp3_file()
    if not audio_path:
        return

    output_path = f"{audio_path.split('.')[0]}_vhs.mp4"
    print(f"Visualizing {audio_path} with {image_path}...")
    print(f"Output will be saved as {output_path}")

    visualizer = VHSMusicVisualizer(image_path, audio_path, output_path)
    visualizer.create_visualization()

    print("Done! Your VHS style visualization is ready.")


if __name__ == "__main__":
    main()
