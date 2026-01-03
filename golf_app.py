import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import tempfile
import pandas as pd
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- 1. 基本設定 ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    video { width: 100% !important; height: auto !important; }
    .compare-metric {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 5px;
        text-align: center;
    }
    .pro-val { color: #2c3e50; }
    .my-val { color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State ---
if 'pro_processed_video' not in st.session_state: st.session_state['pro_processed_video'] = None
if 'pro_df' not in st.session_state: st.session_state['pro_df'] = None
if 'my_processed_video' not in st.session_state: st.session_state['my_processed_video'] = None
if 'my_df' not in st.session_state: st.session_state['my_df'] = None
if 'sync_video_path' not in st.session_state: st.session_state['sync_video_path'] = None
if 'pro_fps' not in st.session_state: st.session_state['pro_fps'] = 30
if 'my_fps' not in st.session_state: st.session_state['my_fps'] = 30

# --- 2. 計算・解析用関数 ---

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def analyze_video(input_path, output_path, rotate_mode="なし"):
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if rotate_mode == "時計回りに90度" or rotate_mode == "反時計回りに90度":
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_data = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        bar = st.progress(0)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            
            if rotate_mode == "時計回りに90度":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_mode == "反時計回りに90度":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow    = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist    = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                nose       = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                
                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                
                pose_data.append({
                    "Frame": i,
                    "Time": i / fps if fps > 0 else 0,
                    "Arm_Angle": angle,
                    "L_Wrist_Y": l_wrist[1], 
                    "Nose_X": nose[0] 
                })

                color = (0, 255, 0) if angle > 160 else (0, 0, 255)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )
                
                h, w, _ = image.shape
                cv2.circle(image, (int(nose[0]*w), int(nose[1]*h)), 5, (255, 255, 0), -1)

            out.write(image)
            if frame_count > 0: bar.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    df = pd.DataFrame(pose_data)
    return output_path, df, fps

def create_sync_video(pro_path, my_path, pro_top_frame, my_top_frame, output_path, target_fps):
    """同期動画生成"""
    cap_pro = cv2.VideoCapture(pro_path)
    cap_my = cv2.VideoCapture(my_path)

    h_pro = int(cap_pro.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_my = int(cap_my.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 読み込み失敗回避
    if h_pro == 0 or h_my == 0:
        return

    target_h = min(h_pro, h_my)
    
    w_pro = int(cap_pro.get(cv2.CAP_PROP_FRAME_WIDTH))
    w_my = int(cap_my.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_w_pro = int(w_pro * (target_h / h_pro))
    new_w_my = int(w_my * (target_h / h_my))
    target_w = new_w_pro + new_w_my

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_w, target_h))

    offset = my_top_frame - pro_top_frame
    pro_delay = max(0, offset)
    my_delay = max(0, -offset)
    
    max_frames = int(max(cap_pro.get(cv2.CAP_PROP_FRAME_COUNT) + pro_delay, 
                         cap_my.get(cv2.CAP_PROP_FRAME_COUNT) + my_delay))

    bar = st.progress(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # バグ修正: 初期値を設定しておく
    sync_text = ""

    for i in range(max_frames):
        if i < pro_delay:
            cap_pro.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_pro, frame_pro = cap_pro.read()
            sync_text = "Waiting for Pro..."
        else:
            ret_pro, frame_pro = cap_pro.read()
        
        if i < my_delay:
            cap_my.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_my, frame_my = cap_my.read()
            sync_text = "Waiting for You..."
        else:
            ret_my, frame_my = cap_my.read()
            
        if not ret_pro or not ret_my: break

        frame_pro_resized = cv2.resize(frame_pro, (new_w_pro, target_h))
        frame_my_resized = cv2.resize(frame_my, (new_w_my, target_h))
        concat_frame = cv2.hconcat([frame_pro_resized, frame_my_resized])
        
        if i == (pro_top_frame + pro_delay): sync_text = "TOP MATCH!"
        
        # テキスト描画（sync_textが空でない場合のみ）
        if sync_text:
            cv2.putText(concat_frame, sync_text, (target_w//2 - 100, 50), font, 1, (0,0,0), 4)
            cv2.putText(concat_frame, sync_text, (target_w//2 - 100, 50), font, 1, (0,255,255), 2)
        
        out.write(concat_frame)
        if max_frames > 0: bar.progress((i + 1) / max_frames)

    cap_pro.release()
    cap_my.release()
    out.release()
    return

# --- 3. メインUI (これが抜けていた！) ---
st.title("🏌️ Golf AI Coach - Dual Analyzer")

tab1, tab2, tab3 = st.tabs(["1. Upload & Analyze", "2. Compare & Sync", "3. Realtime Check"])

# --- Tab 1: アップロードと個別解析 ---
with tab1:
    col1, col2 = st.columns(2)
    
    # --- プロ動画 ---
    with col1:
        st.subheader("Professional / Model Video")
        pro_file = st.file_uploader("Upload Pro Video", type=['mp4', 'mov'], key="pro")
        pro_rotate = st.selectbox("Rotation", ["なし", "時計回りに90度", "反時計回りに90度"], key="pro_rot")
        
        if pro_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(pro_file.read())
            
            if st.button("Analyze Pro Video"):
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('Analyzing Pro Video...'):
                    # 解析実行
                    processed_path, df, fps = analyze_video(tfile.name, out_path, pro_rotate)
                    st.session_state['pro_processed_video'] = processed_path
                    st.session_state['pro_df'] = df
                    st.session_state['pro_fps'] = fps
                st.success("Analysis Complete!")

            if st.session_state['pro_processed_video']:
                st.video(st.session_state['pro_processed_video'])
                if st.session_state['pro_df'] is not None:
                    # トップ位置（最も手が上がった瞬間＝手首Yが最小）の検出
                    min_wrist_idx = st.session_state['pro_df']['L_Wrist_Y'].idxmin()
                    top_frame = st.session_state['pro_df'].loc[min_wrist_idx, 'Frame']
                    st.info(f"Detected Top Position Frame: {top_frame}")

    # --- 自分の動画 ---
    with col2:
        st.subheader("Your Swing")
        my_file = st.file_uploader("Upload Your Video", type=['mp4', 'mov'], key="my")
        my_rotate = st.selectbox("Rotation", ["なし", "時計回りに90度", "反時計回りに90度"], key="my_rot")
        
        if my_file:
            tfile_my = tempfile.NamedTemporaryFile(delete=False)
            tfile_my.write(my_file.read())
            
            if st.button("Analyze My Video"):
                out_path_my = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('Analyzing Your Video...'):
                    processed_path_my, df_my, fps_my = analyze_video(tfile_my.name, out_path_my, my_rotate)
                    st.session_state['my_processed_video'] = processed_path_my
                    st.session_state['my_df'] = df_my
                    st.session_state['my_fps'] = fps_my
                st.success("Analysis Complete!")

            if st.session_state['my_processed_video']:
                st.video(st.session_state['my_processed_video'])
                if st.session_state['my_df'] is not None:
                    min_wrist_idx_my = st.session_state['my_df']['L_Wrist_Y'].idxmin()
                    top_frame_my = st.session_state['my_df'].loc[min_wrist_idx_my, 'Frame']
                    st.info(f"Detected Top Position Frame: {top_frame_my}")

# --- Tab 2: 同期比較 ---
with tab2:
    st.header("Sync & Compare")
    
    if st.session_state['pro_df'] is not None and st.session_state['my_df'] is not None:
        # 自動でトップ位置を推定してデフォルト値にする
        default_pro_top = int(st.session_state['pro_df'].loc[st.session_state['pro_df']['L_Wrist_Y'].idxmin(), 'Frame'])
        default_my_top = int(st.session_state['my_df'].loc[st.session_state['my_df']['L_Wrist_Y'].idxmin(), 'Frame'])
        
        c1, c2 = st.columns(2)
        pro_frame_input = c1.number_input("Pro Top Frame", value=default_pro_top)
        my_frame_input = c2.number_input("My Top Frame", value=default_my_top)
        
        if st.button("Create Synced Video"):
            sync_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            # 元動画のパスが必要だが、ここでは簡略化のため解析済み動画を使う（本来は元動画が良い）
            # 注意: session_stateのパスは解析後の動画なので、これを使って同期動画を作る
            with st.spinner('Creating Synced Video...'):
                create_sync_video(
                    st.session_state['pro_processed_video'], 
                    st.session_state['my_processed_video'], 
                    pro_frame_input, 
                    my_frame_input, 
                    sync_out,
                    st.session_state['my_fps']
                )
                st.session_state['sync_video_path'] = sync_out
            st.success("Synced Video Created!")
            
        if st.session_state['sync_video_path']:
            st.video(st.session_state['sync_video_path'])

    else:
        st.warning("Please analyze both videos in Tab 1 first.")

# --- Tab 3: リアルタイム (WebRTC) ---
with tab3:
    st.header("Realtime Mirror")
    st.write("Camera check...")
    
    webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
