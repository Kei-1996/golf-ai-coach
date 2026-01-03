import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import tempfile
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- 1. 基本設定 ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    video { width: 100% !important; height: auto !important; }
    /* 数値比較を見やすく */
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
    """
    動画解析
    rotate_mode: "なし", "時計回りに90度", "反時計回りに90度"
    """
    cap = cv2.VideoCapture(input_path)
    
    # 元のサイズ取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 回転後のサイズを計算
    if rotate_mode == "時計回りに90度" or rotate_mode == "反時計回りに90度":
        out_width, out_height = height, width # 入れ替え
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
            
            # --- 回転処理 ---
            if rotate_mode == "時計回りに90度":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_mode == "反時計回りに90度":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # ----------------
            
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
                
                # 頭マーカー
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
    
    # フォント設定
    font = cv2.FONT_HERSHEY_SIMPLEX

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
        
        # テキストに影をつけて見やすくする
        cv2.putText(concat_frame, sync_text, (target_w//2 - 100, 50), font, 1, (0,0,0), 4)
        cv2.putText(concat_frame, sync_text, (target_w//2 - 100, 50), font, 1, (0,255,255), 2)
        
        out.write(concat_frame)
        bar.progress((i + 1) / max_frames)

    cap_pro.release()
    cap_my.release()
    out.release()
    return
