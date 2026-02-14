import streamlit as st
from PIL import Image
from ultralytics import YOLO
import random
from pillow_heif import register_heif_opener

# HEIC形式の画像をPILで開けるようにするおまじない
register_heif_opener()

st.set_page_config(page_title="YOLO Quiz App", layout="wide")

st.title("YOLO Low Confidence Quiz (頼りない画像認識クイズ)")
st.caption("画像をアップロードした後、下に表示されるボタンを押すとクイズが始まります！")

# --- Model Loading ---
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- Session State Initialization ---
if 'quiz_active' not in st.session_state:
    st.session_state.quiz_active = False
# 画像が変わったときにリセット判定を行うためのキー
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
# 【追加】ユーザーが選んだ回答を保持する
if 'user_selected_answer' not in st.session_state:
    st.session_state.user_selected_answer = None

# --- Sidebar Settings ---
st.sidebar.title("Settings")
confidence = st.sidebar.slider("判定の厳しさ(Confidence)", 0.0, 1.0, 0.25)

# --- Image Input Section ---
st.subheader("1. 画像を用意する")
input_method = st.radio("入力方法を選択", ["ファイルアップロード", "カメラで撮影"], horizontal=True)

image_source = None

if input_method == "ファイルアップロード":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "heic"])
    if uploaded_file:
        image_source = uploaded_file

elif input_method == "カメラで撮影":
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        image_source = camera_file

# --- Main Logic ---
if image_source is not None:
    # 画像が変わったらクイズの状態をリセット
    if st.session_state.last_uploaded_file != image_source.name:
        st.session_state.quiz_active = False
        st.session_state.user_selected_answer = None # 回答状態もリセット
        st.session_state.last_uploaded_file = image_source.name

    try:
        image = Image.open(image_source)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Target Image", use_container_width=True)

        if not st.session_state.quiz_active:
            if st.button("この画像でクイズを作る！", use_container_width=True):
                
                with st.spinner("AIが画像を解析中..."):
                    results = model(image, conf=confidence)
                
                boxes = results[0].boxes
                
                if len(boxes) == 0:
                    st.warning("何も検出されませんでした。判定を緩くするか、別の画像を試してください！")
                else:
                    # --- Logic to find the "Low Confidence" Object ---
                    class_counts = {}
                    detected_items = []
                    names = results[0].names

                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = box.conf[0]
                        name = names[cls_id]
                        class_counts[name] = class_counts.get(name, 0) + 1
                        detected_items.append({
                            "name": name,
                            "conf": conf,
                            "box": box.xyxy[0].cpu().numpy()
                        })

                    min_count = min(class_counts.values())
                    rare_classes = [name for name, count in class_counts.items() if count == min_count]
                    candidates = [item for item in detected_items if item["name"] in rare_classes]
                    
                    target = sorted(candidates, key=lambda x: x["conf"])[0]

                    x1, y1, x2, y2 = map(int, target["box"])
                    st.session_state.cropped_img = image.crop((x1, y1, x2, y2))
                    st.session_state.correct_answer = target["name"]
                    st.session_state.res_plotted = results[0].plot()

                    detected_names_unique = list(class_counts.keys())
                    wrong_choices = [name for name in detected_names_unique if name != target["name"]]

                    if len(wrong_choices) < 2:
                        all_names = list(names.values())
                        remaining_names = [n for n in all_names if n != target["name"] and n not in wrong_choices]
                        sample_size = min(2, len(remaining_names))
                        if sample_size > 0:
                            wrong_choices += random.sample(remaining_names, sample_size)

                    num_choices = min(2, len(wrong_choices))
                    choices = [target["name"]] + random.sample(wrong_choices, num_choices)
                    random.shuffle(choices)

                    st.session_state.choices = choices
                    st.session_state.quiz_active = True
                    st.session_state.user_selected_answer = None # 新しいクイズの時は回答を空にする
                    st.rerun()

    with col2:
        if st.session_state.quiz_active:
            st.divider()
            st.subheader("Quiz Challenge!")
            st.info("Q. AIはこの切り抜かれた部分を**何**と判定したでしょう？")

            if st.session_state.cropped_img:
                st.image(st.session_state.cropped_img, caption="Target Object", width=200)
            
            # --- 選択肢ボタンの表示エリア ---
            # ここでは「ボタンが押された」という事実だけを保存し、表示は変えない
            for choice in st.session_state.choices:
                # すでに回答済みの場合はボタンを押せないようにする（オプション）
                disabled = st.session_state.user_selected_answer is not None
                
                if st.button(choice, use_container_width=True, disabled=disabled):
                    st.session_state.user_selected_answer = choice
                    st.rerun() # 状態を更新して即再描画

            # --- 結果発表エリア（ループの外に出しました） ---
            if st.session_state.user_selected_answer is not None:
                st.divider() # 線を引いて区切りをわかりやすく
                
                user_ans = st.session_state.user_selected_answer
                correct_ans = st.session_state.correct_answer

                if user_ans == correct_ans:
                    st.success(f"正解！ AIはこれを**{user_ans}**だと判定しました！")
                    st.balloons()
                else:
                    st.error(f"残念... 正解は**{correct_ans}**でした。（あなたの回答: {user_ans}）")
                
                # 正解画像の表示
                if 'res_plotted' in st.session_state:
                    st.image(st.session_state.res_plotted, caption="AI's Answer Key", channels="BGR", use_container_width=True)
                
                # 次へボタン
                if st.button("次の画像へ", type="primary"):
                        st.session_state.quiz_active = False
                        st.session_state.user_selected_answer = None
                        st.rerun()