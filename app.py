"""CrackVision: Concrete Crack Detection web app (Streamlit).

วางไฟล์โมเดล (.keras) และ config.json ที่ได้จาก Colab ไว้ในโฟลเดอร์เดียวกับไฟล์นี้
"""
import json
import os
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

st.set_page_config(page_title="CrackVision", page_icon="🧱", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["No Crack", "Crack"]


# ---------------------------------------------------------------- model
def preprocess(x, model_name):
    """ต้องตรงกับที่ใช้ตอนเทรนใน Colab"""
    if model_name == "mobilenetv2":
        return x / 127.5 - 1.0
    if model_name == "resnet50":  # caffe style: RGB -> BGR แล้วลบค่าเฉลี่ย
        return x[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype="float32")
    return x / 255.0  # cnn_baseline


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model():
    import keras

    with open(os.path.join(APP_DIR, "config.json")) as f:
        cfg = json.load(f)
    model = keras.models.load_model(os.path.join(APP_DIR, cfg["model_file"]), compile=False)
    return model, cfg


def make_patches(img, patch, max_side):
    """ย่อภาพให้ไม่ใหญ่เกิน max_side แล้วตัดเป็นช่องขนาด patch x patch"""
    img = img.copy()
    img.thumbnail((max_side, max_side))
    if min(img.size) < patch:
        s = patch / min(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)))
    arr = np.asarray(img).astype("float32")
    H, W = arr.shape[:2]
    ys = sorted(set(list(range(0, H - patch + 1, patch)) + [H - patch]))  # ช่องสุดท้ายชิดขอบ
    xs = sorted(set(list(range(0, W - patch + 1, patch)) + [W - patch]))  # ให้ครอบทั้งภาพ
    coords = [(y, x) for y in ys for x in xs]
    return img, arr, coords


def predict_patches(model, cfg, arr, coords, patch):
    import tensorflow as tf

    size = cfg["img_size"]
    batch = np.stack(
        [tf.image.resize(arr[y:y + patch, x:x + patch], (size, size)).numpy() for y, x in coords]
    )
    x = preprocess(batch, cfg["model_name"])
    return model.predict(x, batch_size=32, verbose=0).ravel()


def grad_cam(model, cfg, patch_arr):
    """Heatmap ของ 1 ช่อง (ค่า 0-1) บอกว่าโมเดลดูตรงไหนตอนตัดสินว่าร้าว"""
    import keras
    import tensorflow as tf

    size = cfg["img_size"]
    x = tf.image.resize(patch_arr, (size, size)).numpy()[None, ...]
    x = tf.convert_to_tensor(preprocess(x, cfg["model_name"]))
    grad_model = keras.Model(model.inputs, [model.get_layer(cfg["last_conv"]).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(x)
        score = pred[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], patch_arr.shape[:2]).numpy()[..., 0]
    return cam


# ---------------------------------------------------------------- drawing
def draw_boxes(img, coords, probs, patch, threshold):
    out = img.convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = ImageFont.load_default(size=max(14, patch // 10))
    for (y, x), p in zip(coords, probs):
        if p >= threshold:
            d.rectangle([x, y, x + patch - 1, y + patch - 1], fill=(255, 0, 0, 60),
                        outline=(255, 0, 0, 255), width=3)
            label = f"{p:.0%}"
            l, t, r, b = d.textbbox((x + 8, y + 6), label, font=font)
            d.rectangle([l - 5, t - 4, r + 5, b + 4], fill=(220, 0, 0, 230))
            d.text((x + 8, y + 6), label, fill=(255, 255, 255, 255), font=font)
        else:
            d.rectangle([x, y, x + patch - 1, y + patch - 1], outline=(255, 255, 255, 70), width=1)
    return Image.alpha_composite(out, layer).convert("RGB")


def jet(v):
    """colormap แบบ jet (ไม่ต้องพึ่ง matplotlib)"""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def heatmap_overlay(img, arr, coords, probs, patch, threshold, model, cfg):
    heat = np.zeros(arr.shape[:2], dtype="float32")
    for (y, x), p in zip(coords, probs):
        if p >= threshold:
            heat[y:y + patch, x:x + patch] = grad_cam(model, cfg, arr[y:y + patch, x:x + patch]) * p
    base = np.asarray(img).astype("float32") / 255.0
    alpha = (heat * 0.6)[..., None]
    out = base * (1 - alpha) + jet(heat) * alpha
    return Image.fromarray((out * 255).astype("uint8"))


# ---------------------------------------------------------------- UI
st.title("🧱 CrackVision")
st.caption("ระบบตรวจจับรอยร้าวคอนกรีตจากภาพถ่าย · Concrete Crack Detection for Building Inspection")

try:
    model, cfg = load_model()
except Exception as e:  # แสดงข้อความชัด ๆ ถ้าลืมอัปโหลดไฟล์โมเดล
    st.error(f"โหลดโมเดลไม่สำเร็จ: ตรวจว่ามีไฟล์ config.json และไฟล์ .keras อยู่ในโฟลเดอร์เดียวกับ app.py\n\n{e}")
    st.stop()

with st.sidebar:
    st.header("ตั้งค่า")
    threshold = st.slider("Threshold (ค่าความมั่นใจขั้นต่ำ)", 0.05, 0.95,
                          float(cfg.get("threshold", 0.5)), 0.05,
                          help="ลดลง = หารอยร้าวเจอมากขึ้น (Recall สูง) แต่เตือนผิดมากขึ้น")
    max_side = st.select_slider("ความละเอียดในการสแกน (ด้านยาวสุด, px)",
                                options=[454, 681, 908, 1135, 1362], value=908,
                                help="ยิ่งมาก ช่องยิ่งเล็กเมื่อเทียบกับภาพ เหมาะกับรอยร้าวเล็ก แต่ช้าลง")
    show_cam = st.checkbox("แสดง Grad-CAM", value=True)
    st.divider()
    st.markdown(
        f"**Model:** `{cfg['model_name']}`  \n"
        f"**Patch:** {cfg['patch']}×{cfg['patch']} px"
    )
    st.markdown(
        "**วิธีทำงาน**\n"
        "1. ย่อภาพและตัดเป็นช่องย่อย\n"
        "2. โมเดลทำนายทีละช่องว่ามีรอยร้าวไหม\n"
        "3. ระบายสีแดงช่องที่ความน่าจะเป็น ≥ threshold\n"
        "4. Grad-CAM ชี้ตำแหน่งที่โมเดลใช้ตัดสิน"
    )

tab_up, tab_cam = st.tabs(["📁 อัปโหลดภาพ", "📷 ถ่ายด้วยกล้อง"])
with tab_up:
    up = st.file_uploader("เลือกภาพพื้นผิวคอนกรีต (JPG/PNG)", type=["jpg", "jpeg", "png"])
with tab_cam:
    shot = st.camera_input("ถ่ายภาพผนังหรือพื้น")

source = up or shot
if source is None:
    st.info("อัปโหลดภาพหรือถ่ายภาพเพื่อเริ่มตรวจ · แนะนำถ่ายห่างพื้นผิวประมาณ 30–60 ซม.")
    st.stop()

img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
patch = int(cfg.get("patch", 227))

t0 = time.time()
small, arr, coords = make_patches(img, patch, max_side)
probs = predict_patches(model, cfg, arr, coords, patch)
elapsed = time.time() - t0

n_crack = int((probs >= threshold).sum())
has_crack = n_crack > 0
max_p = float(probs.max())

if has_crack:
    st.error(f"### ⚠️ พบรอยร้าว (Crack detected) · {n_crack} จาก {len(coords)} ช่อง")
else:
    st.success(f"### ✅ ไม่พบรอยร้าว (No crack) · ตรวจ {len(coords)} ช่อง")

c1, c2, c3 = st.columns(3)
c1.metric("ผลรวม", CLASS_NAMES[int(has_crack)])
c2.metric("ความมั่นใจสูงสุดว่าร้าว", f"{max_p:.1%}")
c3.metric("เวลาประมวลผล", f"{elapsed:.2f} s")

boxed = draw_boxes(small, coords, probs, patch, threshold)
if show_cam and has_crack:
    with st.spinner("กำลังสร้าง Grad-CAM..."):
        cam_img = heatmap_overlay(small, arr, coords, probs, patch, threshold, model, cfg)
    col1, col2 = st.columns(2)
    col1.image(boxed, caption="ช่องที่พบรอยร้าว (กรอบแดง + % ความมั่นใจ)", use_container_width=True)
    col2.image(cam_img, caption="Grad-CAM: บริเวณที่โมเดลใช้ตัดสิน", use_container_width=True)
else:
    st.image(boxed, caption="ผลการสแกน (กรอบแดง = มีรอยร้าว)", use_container_width=True)

with st.expander("ดูความน่าจะเป็นของแต่ละช่อง"):
    rows = sorted(zip(coords, probs), key=lambda t: -t[1])
    st.dataframe(
        [{"y": y, "x": x, "crack prob": round(float(p), 4),
          "result": CLASS_NAMES[int(p >= threshold)]} for (y, x), p in rows],
        use_container_width=True, hide_index=True,
    )

st.caption("โมเดลเทรนด้วย Concrete Crack Images for Classification (Özgenel, 2018) · "
           "ผลลัพธ์ใช้ช่วยคัดกรองเบื้องต้น ไม่ใช่การประเมินโครงสร้างโดยวิศวกร")
