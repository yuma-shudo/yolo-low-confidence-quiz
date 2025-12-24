import streamlit as st
from PIL import Image
from ultralytics import YOLO
import random

st.set_page_config(page_title="YOLO Quiz App", layout="wide")

st.title("YOLO Object Detection & Quiz App")
st.subheader("画像をアップロードした後、下に表示されるボタンを押すとクイズが始まります！")

#重いのでcacheを使ってmodelをロード
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

#中断しても続きから始められるようsession_stateを用意
if 'quiz_active' not in st.session_state:
    st.session_state.quiz_active = False
if 'target_label' not in st.session_state:
    st.session_state.target_label = ""
if 'choices' not in st.session_state:
    st.session_state.choices = []
if 'correct_answer' not in st.session_state:
    st.session_state.correct_answer = ""
if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None

#信頼度の下限を決めるsliderの設定
st.sidebar.title("Settings")
confidence = st.sidebar.slider("判定の厳しさ(Confidence)", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader("Upload an image to generate Quiz!", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    #元の画像を示す場所とクイズを示す場所で左右に分割
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original Image")

        results = model(image, conf=confidence)

        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("この画像でクイズを作る！"):
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.error("何も検出されませんでした。別の画像をお試しください！") 
        else:
            class_counts = {}
            detected_items = []

            names = results[0].names #class_idとnameの対応表

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = box.conf[0]
                name = names[cls_id]

                #未登録のキーでもエラーを起こさないようにget関数で例外を指定する
                class_counts[name] = class_counts.get(name, 0) + 1

                detected_items.append({
                    "name": name,
                    "conf": conf,
                    #普通のコンピューターで扱えるようにcpuでの処理、テンソルをnumpy配列に変換
                    "box": box.xyxy[0].cpu().numpy()
                })
            
            min_count = min(class_counts.values())
            #一番少ないオブジェクトの名称を取得する
            rare_classes = [name for name, count in class_counts.items() if count == min_count]
            #名称を手掛かりに、信頼度と座標も併せて取得する
            candidates = [item for item in detected_items if item["name"] in rare_classes]

            #一番少ないオブジェクトの中でも、一番信頼度が低いものをクイズにする
            target = sorted(candidates, key=lambda x: x["conf"])[0]

            x1, y1, x2, y2 = map(int, target["box"])
            st.session_state.cropped_img = image.crop((x1, y1, x2, y2))
            st.session_state.correct_answer = target["name"]

            detected_names_unique = list(class_counts.keys())

            wrong_choices = [name for name in detected_names_unique if name != target["name"]]

            #選択肢が足りない時は全体から取得する
            if len(wrong_choices) < 2:
                all_names = list(names.values())
                remaining_names = [n for n in all_names if n != target["name"] and n not in wrong_choices]
                wrong_choices += random.sample(remaining_names, min(2, len(remaining_names)))

            choices = [target["name"]] + random.sample(wrong_choices, min(2, len(wrong_choices)))
            random.shuffle(choices)

            st.session_state.choices = choices
            st.session_state.quiz_active = True

            st.rerun()

    with col2:
        if st.session_state.quiz_active:
            st.subheader("Quiz Challenge!")
            st.write("Q. AIはこの切り抜かれた部分を**何**と判定したでしょう？")

            if st.session_state.cropped_img:
                st.image(st.session_state.cropped_img, caption="Target Object", width=200)
            
            for choice in st.session_state.choices:
                if st.button(choice):
                    if choice == st.session_state.correct_answer:
                        st.success(f"正解！ AIはこれを**{choice}**だと判定しました！")
                        st.balloons()
                    else:
                        st.error(f"残念... 正解は**{st.session_state.correct_answer}**でした。")

                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption="AI's Answer Key", channels="BGR", use_container_width=True)

                    st.session_state.quiz_active = False