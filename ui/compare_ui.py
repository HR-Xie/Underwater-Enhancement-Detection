import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image
import torch
import os
import sys
import threading
import numpy as np
import subprocess
import time

# Add project root and detection module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'detection'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from WIModel import LightAquaNet

# --------------------------
# 1. 设备与模型加载
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# Resolve paths relative to this script's location
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENHANCE_WEIGHT = os.path.join(_BASE_DIR, '..', 'weights', 'UW_epoch_277.pth')
_DETECT_WEIGHT = os.path.join(_BASE_DIR, '..', 'weights', 'best.pt')


def load_models():
    print("正在加载模型，请稍候...")
    water_inhance_model = LightAquaNet().to(device)
    checkpoint = torch.load(_ENHANCE_WEIGHT, map_location=device)
    water_inhance_model.load_state_dict(checkpoint['state_dict'])
    water_inhance_model.eval()

    from ultralytics import YOLO
    if not os.path.exists(_DETECT_WEIGHT):
        raise FileNotFoundError(
            f"Detection weight not found: {_DETECT_WEIGHT}\n"
            f"Please place your trained detection model (best.pt) in the weights/ directory."
        )
    yolo_model = YOLO(_DETECT_WEIGHT)

    print("模型加载完成！")
    return water_inhance_model, yolo_model


# --------------------------
# 2. 核心处理与统计算法
# --------------------------
def process_frame(frame, use_enhance, use_detect, enhance_model, yolo_model, conf_threshold):
    """处理单帧图像，返回处理后的画面以及检测统计数据"""
    stats = {}

    # --- 1. 图像增强 ---
    if use_enhance and enhance_model is not None:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transposed_img = np.ascontiguousarray(img_rgb.transpose((2, 0, 1)))

        input_tensor = torch.from_numpy(transposed_img).to(device, dtype=torch.float32) / 255.0
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output_tensor = enhance_model(input_tensor)
            if isinstance(output_tensor, tuple) or isinstance(output_tensor, list):
                output_tensor = output_tensor[0]

        output_img = output_tensor.squeeze().cpu().clamp(0, 1).numpy()
        output_img = np.ascontiguousarray(np.transpose(output_img, (1, 2, 0)))
        output_img = (output_img * 255.0).astype(np.uint8)
        frame = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    # --- 2. 目标检测 ---
    if use_detect and yolo_model is not None:
        results = yolo_model(frame, device=device, conf=conf_threshold, verbose=False)
        frame = results[0].plot()

        names = results[0].names
        for cls_id in results[0].boxes.cls:
            class_name = names[int(cls_id)]
            stats[class_name] = stats.get(class_name, 0) + 1

    return frame, stats


def get_fit_size(orig_w, orig_h, max_w, max_h):
    ratio = min(max_w / orig_w, max_h / orig_h)
    return int(orig_w * ratio), int(orig_h * ratio)


# --------------------------
# 3. 三栏全景旗舰 GUI 界面类
# --------------------------
class UnderwaterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("水下视觉智能分析系统 (旗舰展示版)")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.file_path = None
        self.save_dir = os.getcwd()
        self.output_file_path = ""

        self.is_enhance_on = False
        self.is_detect_on = False

        self.main_bg = "#191919"
        self.card_bg = "#252525"
        self.accent_blue = "#007AFF"
        self.accent_green = "#28a745"
        self.accent_gray = "#444444"

        self.ctk_orig = None
        self.ctk_proc = None
        empty_pil = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        self.empty_ctk_image = ctk.CTkImage(light_image=empty_pil, size=(1, 1))

        self.configure(fg_color=self.main_bg)
        self.enhance_model, self.yolo_model = load_models()

        self.class_name_map = {
            "holothurian": "海参",
            "echinus": "海胆",
            "scallop": "扇贝",
            "starfish": "海星"
        }

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=320)
        self.grid_rowconfigure(0, weight=1)

        # ================= 左侧控制面板 =================
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=25, fg_color=self.card_bg, border_width=1,
                                          border_color="#333")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        self.sidebar_frame.grid_propagate(False)

        self.bottom_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.bottom_frame.pack(side="bottom", fill="x", padx=15, pady=(0, 25))
        self.progressbar = ctk.CTkProgressBar(self.bottom_frame, height=5, progress_color=self.accent_blue)
        self.progressbar.pack(fill="x", pady=(0, 10))
        self.progressbar.set(0)
        self.start_btn = ctk.CTkButton(self.bottom_frame, text="▶ 开始执行分析",
                                       font=ctk.CTkFont(size=16, weight="bold"), height=45, fg_color=self.accent_green,
                                       hover_color="#218838", corner_radius=22, command=self.start_processing_thread)
        self.start_btn.pack(fill="x")
        self.status_label = ctk.CTkLabel(self.bottom_frame, text="就绪", text_color="gray", font=ctk.CTkFont(size=11))
        self.status_label.pack(pady=(5, 0))

        self.title_container = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.title_container.pack(pady=(25, 15))
        ctk.CTkLabel(self.title_container, text="系统控制台", font=ctk.CTkFont(size=22, weight="bold")).pack()
        ctk.CTkLabel(self.title_container, text="UVAIS 动态交互版", font=ctk.CTkFont(size=11),
                     text_color="gray60").pack()

        self.create_card_title(self.sidebar_frame, "📂 文件源载入")
        self.input_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#2c2c2c", corner_radius=15, border_width=1,
                                        border_color="#3a3a3a")
        self.input_frame.pack(fill="x", padx=15, pady=(0, 15))
        self.upload_btn = ctk.CTkButton(self.input_frame, text="选择视频/图像...",
                                        font=ctk.CTkFont(weight="bold", size=13), height=36, corner_radius=18,
                                        command=self.upload_file)
        self.upload_btn.pack(pady=(15, 5), padx=15, fill="x")
        self.file_label = ctk.CTkLabel(self.input_frame, text="等待加载...", text_color="gray", wraplength=260,
                                       font=ctk.CTkFont(size=11))
        self.file_label.pack(pady=(0, 15), padx=15)

        self.create_card_title(self.sidebar_frame, "⚙️ 算法引擎 (实时控制)")
        self.func_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#2c2c2c", corner_radius=15, border_width=1,
                                       border_color="#3a3a3a")
        self.func_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.btn_toggle_enhance = ctk.CTkButton(self.func_frame, text="图像增强 (LightAquaNet): 已关闭",
                                                fg_color=self.accent_gray, hover_color="#555",
                                                command=self.toggle_enhance)
        self.btn_toggle_enhance.pack(pady=(15, 5), padx=15, fill="x")

        self.btn_toggle_detect = ctk.CTkButton(self.func_frame, text="目标检测 (SEAD-YOLO): 已关闭",
                                               fg_color=self.accent_gray, hover_color="#555",
                                               command=self.toggle_detect)
        self.btn_toggle_detect.pack(pady=(5, 15), padx=15, fill="x")

        self.create_card_title(self.sidebar_frame, "💾 输出目录")
        self.output_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#2c2c2c", corner_radius=15, border_width=1,
                                         border_color="#3a3a3a")
        self.output_frame.pack(fill="x", padx=15, pady=(0, 10))
        self.save_dir_btn = ctk.CTkButton(self.output_frame, text="更改目录...", fg_color="transparent", border_width=1,
                                          border_color="#555", text_color="#DCE4EE", corner_radius=16,
                                          command=self.choose_save_dir)
        self.save_dir_btn.pack(pady=(12, 5), padx=12, fill="x")
        self.save_dir_label = ctk.CTkLabel(self.output_frame, text=f"{self.save_dir}", text_color="gray",
                                           wraplength=260, font=ctk.CTkFont(size=10))
        self.save_dir_label.pack(pady=(0, 10), padx=12)

        # ================= 中间主视窗区 =================
        self.main_center_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_center_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=20)
        self.main_center_frame.grid_rowconfigure(0, weight=1, uniform="row_group")
        self.main_center_frame.grid_rowconfigure(1, weight=1, uniform="row_group")
        self.main_center_frame.grid_columnconfigure(0, weight=1)

        self.frame_top = ctk.CTkFrame(self.main_center_frame, corner_radius=20, fg_color=self.card_bg, border_width=1,
                                      border_color="#333")
        self.frame_top.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.create_stream_label(self.frame_top, "🎥 原始画面输入流", "gray80").pack(pady=(10, 0), anchor="w", padx=20)
        self.label_orig = ctk.CTkLabel(self.frame_top, text="等待输入...", text_color="#333",
                                       font=ctk.CTkFont(size=22, weight="bold"))
        self.label_orig.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.frame_bottom = ctk.CTkFrame(self.main_center_frame, corner_radius=20, fg_color=self.card_bg,
                                         border_width=1, border_color="#333")
        self.frame_bottom.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.create_stream_label(self.frame_bottom, "📺 算法综合推理流", self.accent_blue).pack(pady=(10, 0), anchor="w",
                                                                                               padx=20)
        self.label_proc = ctk.CTkLabel(self.frame_bottom, text="等待处理...", text_color="#333",
                                       font=ctk.CTkFont(size=22, weight="bold"))
        self.label_proc.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # ================= 右侧实时数据看板 =================
        self.dashboard_frame = ctk.CTkFrame(self, corner_radius=25, fg_color=self.card_bg, border_width=1,
                                            border_color="#333")
        self.dashboard_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 20), pady=20)

        ctk.CTkLabel(self.dashboard_frame, text="📊 实时智能分析看板", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=self.accent_blue).pack(pady=(25, 20))

        self.create_card_title(self.dashboard_frame, "🎯 检测置信度阈值 (可实时拖拽)")
        self.param_frame = ctk.CTkFrame(self.dashboard_frame, fg_color="#2c2c2c", corner_radius=15)
        self.param_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.conf_val_label = ctk.CTkLabel(self.param_frame, text="当前阈值: 0.25",
                                           font=ctk.CTkFont(size=13, weight="bold"), text_color="#00adb5")
        self.conf_val_label.pack(pady=(10, 0))

        self.conf_slider = ctk.CTkSlider(self.param_frame, from_=0.01, to=0.99, command=self.slider_event)
        self.conf_slider.set(0.25)
        self.conf_slider.pack(fill="x", padx=20, pady=(10, 15))

        self.create_card_title(self.dashboard_frame, "📌 当前帧检测统计")
        self.stats_frame = ctk.CTkFrame(self.dashboard_frame, fg_color="#2c2c2c", corner_radius=15)
        self.stats_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.stats_textbox = ctk.CTkTextbox(self.stats_frame, fg_color="transparent", font=ctk.CTkFont(size=14),
                                            text_color="white", wrap="word")
        self.stats_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        self.stats_textbox.insert("0.0", "暂无检测数据...\n(请开启目标检测引擎)")
        self.stats_textbox.configure(state="disabled")

        self.create_card_title(self.dashboard_frame, "⚡ 性能监控")
        self.perf_frame = ctk.CTkFrame(self.dashboard_frame, fg_color="#2c2c2c", corner_radius=15)
        self.perf_frame.pack(fill="x", padx=15, pady=(0, 20))

        self.latency_label = ctk.CTkLabel(self.perf_frame, text="单帧推理延迟: -- ms",
                                          font=ctk.CTkFont(size=13, family="Consolas"))
        self.latency_label.pack(anchor="w", padx=15, pady=(15, 5))
        self.fps_label = ctk.CTkLabel(self.perf_frame, text="实时处理帧率: -- FPS",
                                      font=ctk.CTkFont(size=13, family="Consolas"))
        self.fps_label.pack(anchor="w", padx=15, pady=(0, 15))

    # ================= UI 交互逻辑 =================
    def create_card_title(self, parent, title):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="gray60").pack(
            pady=(0, 5), anchor="w", padx=20)

    def create_stream_label(self, parent, title, color):
        return ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(weight="bold", size=14), text_color=color)

    def slider_event(self, value):
        self.conf_val_label.configure(text=f"当前阈值: {value:.2f}")

    def toggle_enhance(self):
        self.is_enhance_on = not self.is_enhance_on
        if self.is_enhance_on:
            self.btn_toggle_enhance.configure(text="图像增强 (LightAquaNet): 已开启", fg_color=self.accent_blue)
        else:
            self.btn_toggle_enhance.configure(text="图像增强 (LightAquaNet): 已关闭", fg_color=self.accent_gray)

    def toggle_detect(self):
        self.is_detect_on = not self.is_detect_on
        if self.is_detect_on:
            self.btn_toggle_detect.configure(text="目标检测 (YOLO): 已开启", fg_color="#ff9f43")
        else:
            self.btn_toggle_detect.configure(text="目标检测 (YOLO): 已关闭", fg_color=self.accent_gray)

    def choose_save_dir(self):
        dir_path = filedialog.askdirectory(title="选择保存目录")
        if dir_path:
            self.save_dir = dir_path
            self.save_dir_label.configure(text=f"{self.save_dir}")

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[("视频与图像文件", "*.jpg *.jpeg *.png *.mp4 *.avi")]
        )
        if self.file_path:
            filename = os.path.basename(self.file_path)
            self.file_label.configure(text=filename)

            self.label_proc.configure(image=self.empty_ctk_image, text="✅ 引擎就绪，等待启动...")

            if self.file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_bgr = cv2.imread(self.file_path)
                h, w = img_bgr.shape[:2]
                new_w, new_h = get_fit_size(w, h, max_w=850, max_h=320)
                img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                self.update_dual_display(orig_rgb=img_rgb, proc_rgb=None, w=new_w, h=new_h)
            else:
                self.label_orig.configure(image=self.empty_ctk_image, text="🎬 视频流已挂载\n开启所需引擎并点击开始")

            self.status_label.configure(text="文件已加载", text_color="#00adb5")
            self.progressbar.set(0)

    # ================= 视图与数据看板更新 =================
    def update_progress(self, value, text, color="orange"):
        self.progressbar.set(value)
        self.status_label.configure(text=text, text_color=color)

    def update_dual_display(self, orig_rgb, proc_rgb, w, h):
        if orig_rgb is not None:
            img_orig = Image.fromarray(orig_rgb)
            self.ctk_orig = ctk.CTkImage(light_image=img_orig, size=(w, h))
            self.label_orig.configure(image=self.ctk_orig, text="")

        if proc_rgb is not None:
            img_proc = Image.fromarray(proc_rgb)
            self.ctk_proc = ctk.CTkImage(light_image=img_proc, size=(w, h))
            self.label_proc.configure(image=self.ctk_proc, text="")

    def update_dashboard(self, stats, latency_ms):
        self.stats_textbox.configure(state="normal")
        self.stats_textbox.delete("0.0", "end")

        if not self.is_detect_on:
            self.stats_textbox.insert("0.0", "目标检测引擎已休眠 Zzz...\n请在左侧面板开启。")
        elif not stats:
            self.stats_textbox.insert("0.0", "当前帧未发现目标...")
        else:
            display_text = ""
            total_objs = 0
            for en_name, count in stats.items():
                cn_name = self.class_name_map.get(en_name, en_name)
                display_text += f"🔹 {cn_name}: {count} 只\n"
                total_objs += count
            display_text = f"总计发现: {total_objs} 个目标\n{'-' * 20}\n" + display_text
            self.stats_textbox.insert("0.0", display_text)

        self.stats_textbox.configure(state="disabled")

        self.latency_label.configure(text=f"单帧推理延迟: {latency_ms:.1f} ms")
        if latency_ms > 0:
            fps = 1000.0 / latency_ms
            self.fps_label.configure(text=f"实时处理帧率: {fps:.1f} FPS")

    # ================= 后台处理线程 =================
    def start_processing_thread(self):
        if not self.file_path:
            messagebox.showwarning("提示", "请先选择待处理文件！")
            return
        if not self.is_enhance_on and not self.is_detect_on:
            messagebox.showwarning("提示", "请在左侧至少开启一项算法引擎！")
            return

        self.start_btn.configure(state="disabled", text="⏳ 引擎全速运转中...")
        self.upload_btn.configure(state="disabled")
        threading.Thread(target=self.process_media, daemon=True).start()

    def process_media(self):
        cap = None
        out = None
        try:
            is_video = self.file_path.lower().endswith(('.mp4', '.avi'))
            filename = os.path.basename(self.file_path)
            self.output_file_path = os.path.join(self.save_dir, "Processed_" + filename)

            SAFE_MAX_W = 850
            SAFE_MAX_H = 320

            if not is_video:
                img_bgr = cv2.imread(self.file_path)
                self.after(0, self.update_progress, 0.5, "正在推理...", "orange")

                start_time = time.time()
                curr_enhance = self.is_enhance_on
                curr_detect = self.is_detect_on
                curr_conf = self.conf_slider.get()

                result_img, stats = process_frame(img_bgr.copy(), curr_enhance, curr_detect, self.enhance_model,
                                                  self.yolo_model, curr_conf)

                latency_ms = (time.time() - start_time) * 1000
                cv2.imwrite(self.output_file_path, result_img)

                h, w = img_bgr.shape[:2]
                new_w, new_h = get_fit_size(w, h, SAFE_MAX_W, SAFE_MAX_H)
                orig_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                proc_resized = cv2.resize(result_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                self.after(0, self.update_progress, 1.0, "处理完成", "#28a745")
                self.after(0, self.update_dual_display, cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(proc_resized, cv2.COLOR_BGR2RGB), new_w, new_h)
                self.after(0, self.update_dashboard, stats, latency_ms)

            else:
                cap = cv2.VideoCapture(self.file_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_origin = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                new_w, new_h = get_fit_size(width, height, SAFE_MAX_W, SAFE_MAX_H)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(self.output_file_path, fourcc, fps_origin, (width, height))

                scale_ratio = 640.0 / max(width, height)
                infer_w, infer_h = int(width * scale_ratio), int(height * scale_ratio)
                current_frame = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    curr_enhance = self.is_enhance_on
                    curr_detect = self.is_detect_on
                    curr_conf = self.conf_slider.get()

                    start_time = time.time()
                    small_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                    small_result, stats = process_frame(small_frame, curr_enhance, curr_detect, self.enhance_model,
                                                        self.yolo_model, curr_conf)
                    processed_frame = cv2.resize(small_result, (width, height), interpolation=cv2.INTER_LINEAR)
                    latency_ms = (time.time() - start_time) * 1000

                    out.write(processed_frame)
                    current_frame += 1

                    if current_frame % 5 == 0 or current_frame == total_frames:
                        progress_val = current_frame / total_frames
                        orig_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        proc_resized = cv2.resize(processed_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                        self.after(0, self.update_dual_display, cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB),
                                   cv2.cvtColor(proc_resized, cv2.COLOR_BGR2RGB), new_w, new_h)
                        self.after(0, self.update_dashboard, stats, latency_ms)
                        self.after(0, self.update_progress, progress_val,
                                   f"⚡ 视频解析中... {current_frame}/{total_frames}", "orange")

                self.after(0, self.update_progress, 1.0, "视频分析完成", "#28a745")

            self.after(500, self.processing_done)

        except Exception as e:
            print(f"处理出错: {e}")
            self.after(0, self.update_progress, 0, "任务中断", "red")
            self.after(0, self.processing_error, str(e))

        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

    def processing_done(self):
        self.start_btn.configure(state="normal", text="▶ 再次执行分析")
        self.upload_btn.configure(state="normal")
        if messagebox.askyesno("成功", f"文件已保存至:\n{self.output_file_path}\n\n是否打开所在文件夹？"):
            try:
                os.startfile(self.save_dir)
            except AttributeError:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, self.save_dir])

    def processing_error(self, error_msg):
        self.start_btn.configure(state="normal", text="▶ 重新执行")
        self.upload_btn.configure(state="normal")
        messagebox.showerror("运行错误", f"算法推理异常:\n{error_msg}")


if __name__ == "__main__":
    app = UnderwaterApp()
    app.mainloop()
