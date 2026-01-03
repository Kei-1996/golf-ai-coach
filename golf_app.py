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
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center; }
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

def analyze_video(input_path, output_path):
    """動画解析：肘、鼻(頭)、手首Y(高さ)を記録"""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_data = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        bar = st.progress(0)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 座標取得
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow    = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist    = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                nose       = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                
                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                
                pose_data.append({
                    "Frame": i,
                    "Time": i / fps if fps > 0 else 0,
                    "Arm_Angle": angle,
                    "L_Wrist_Y": l_wrist[1], # Y座標: 小さい=高い、大きい=低い
                    "Nose_X": nose[0]        # スウェイ判定用
                })

                # 描画
                color = (0, 255, 0) if angle > 160 else (0, 0, 255)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )
                
                # 頭の位置をマーク
                h, w, _ = image.shape
                cv2.circle(image, (int(nose[0]*w), int(nose[1]*h)), 5, (255, 255, 0), -1)

            out.write(image)
            if frame_count > 0: bar.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    df = pd.DataFrame(pose_data)
    return output_path, df, fps

def create_sync_video(pro_path, my_path, pro_top_frame, my_top_frame, output_path, target_fps):
    """同期動画生成（トップ位置合わせ）"""
    cap_pro = cv2.VideoCapture(pro_path)
    cap_my = cv2.VideoCapture(my_path)

    # 高さ合わせ
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
    sync_text = "Syncing..."

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
        
        if i == (pro_top_frame + pro_delay): sync_text = "TOP POSITION MATCHED!"
        cv2.putText(concat_frame, sync_text, (target_w//2 - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        
        out.write(concat_frame)
        bar.progress((i + 1) / max_frames)

    cap_pro.release()
    cap_my.release()
    out.release()
    return output_path

# --- 3. リアルタイム用 ---
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- 4. アプリメイン ---
st.title("⛳️ K's Golf AI Coach")
st.sidebar.header("メニュー")
app_mode = st.sidebar.selectbox("モードを選択", ["リアルタイム判定 (Real-time)", "動画アップロード分析 (Upload)"])
st.sidebar.divider()
club_select = st.sidebar.selectbox("使用クラブ", ["ドライバー", "アイアン", "ウェッジ", "パター"])

# --- モードA ---
if app_mode == "リアルタイム判定 (Real-time)":
    st.header("⚡️ リアルタイム・コーチ")
    col1, col2 = st.columns(2)
    with col1:
        st.info("👈 プロのお手本")
        st.image("https://via.placeholder.com/360x640.png?text=Pro+Swing", use_container_width=True)
    with col2:
        st.success("📸 カメラ映像")
        webrtc_streamer(key="golf-realtime", mode=WebRtcMode.SENDRECV, video_processor_factory=PoseProcessor)

# --- モードB ---
elif app_mode == "動画アップロード分析 (Upload)":
    st.header("📂 動画分析ラボ")
    st.warning("⚠️ **重要:** 正確な比較のため、**プロの動画と「同じアングル」** で撮影された動画を使用してください。")
    
    col1, col2 = st.columns(2)

    # プロ動画
    with col1:
        st.subheader("1. プロ/お手本の動画")
        pro_video = st.file_uploader("プロの動画", type=['mp4', 'mov'], key="pro_video")
        if pro_video is not None:
            if st.button("🔍 プロ解析"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(pro_video.read())
                with st.spinner("プロ解析中..."):
                    path, df, fps = analyze_video(tfile.name, tfile.name + "_pro.mp4")
                    st.session_state['pro_processed_video'] = path
                    st.session_state['pro_df'] = df
                    st.session_state['pro_fps'] = fps
                    st.success("完了")
            if st.session_state['pro_processed_video']:
                st.video(st.session_state['pro_processed_video'])

    # 自分動画
    with col2:
        st.subheader("2. あなたのスイング動画")
        my_video = st.file_uploader("自分の動画", type=['mp4', 'mov'], key="my_video")
        if my_video is not None:
            if st.button("🚀 自分解析"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(my_video.read())
                with st.spinner("自分解析中..."):
                    path, df, fps = analyze_video(tfile.name, tfile.name + "_my.mp4")
                    st.session_state['my_processed_video'] = path
                    st.session_state['my_df'] = df
                    st.session_state['my_fps'] = fps
                    st.success("完了")
            if st.session_state['my_processed_video']:
                st.video(st.session_state['my_processed_video'])

    # --- 総合評価セクション ---
    if st.session_state['pro_df'] is not None and st.session_state['my_df'] is not None:
        st.divider()
        st.header("📊 総合スイング診断")
        
        pro_df = st.session_state['pro_df']
        my_df = st.session_state['my_df']
        
        # --- 1. トップ検出 (一番手が上がった瞬間) ---
        pro_top_idx = pro_df['L_Wrist_Y'].idxmin()
        my_top_idx = my_df['L_Wrist_Y'].idxmin()
        
        # --- 2. インパクト検出 (トップの後に、手が一番下がった瞬間) ---
        # プロ
        pro_after_top = pro_df.iloc[pro_top_idx:] # トップ以降のデータ
        if not pro_after_top.empty:
            pro_impact_idx = pro_after_top['L_Wrist_Y'].idxmax() # 一番下がった位置(Y最大)
        else:
            pro_impact_idx = pro_top_idx # エラー回避

        # 自分
        my_after_top = my_df.iloc[my_top_idx:]
        if not my_after_top.empty:
            my_impact_idx = my_after_top['L_Wrist_Y'].idxmax()
        else:
            my_impact_idx = my_top_idx

        # --- スコア計算 ---
        
        # ① 肘の角度 (トップ時)
        pro_arm = pro_df.iloc[pro_top_idx]['Arm_Angle']
        my_arm = my_df.iloc[my_top_idx]['Arm_Angle']
        diff_arm = abs(my_arm - pro_arm)
        score_arm = max(0, 100 - diff_arm * 2)

        # ② 頭の安定性 (全期間の標準偏差)
        pro_sway = pro_df['Nose_X'].std() * 100
        my_sway = my_df['Nose_X'].std() * 100
        diff_sway = max(0, my_sway - pro_sway)
        score_sway = max(0, 100 - diff_sway * 10)

        # ③ ダウンスイング・テンポ (トップ〜インパクトの時間)
        pro_down_time = pro_df.iloc[pro_impact_idx]['Time'] - pro_df.iloc[pro_top_idx]['Time']
        my_down_time = my_df.iloc[my_impact_idx]['Time'] - my_df.iloc[my_top_idx]['Time']
        diff_time = abs(my_down_time - pro_down_time)
        # 0.1秒ズレるごとに20点減点 (タイミングはシビアに)
        score_tempo = max(0, 100 - (diff_time * 100 * 2))

        # 総合点
        total_score = int((score_arm + score_sway + score_tempo) / 3)

        # --- 表示 ---
        st.subheader(f"🏆 総合スコア: {total_score}点")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("① トップの形(肘)", f"{int(score_arm)}点", f"角度差: {int(diff_arm)}°")
        c2.metric("② 頭の安定性", f"{int(score_sway)}点", f"ブレ差: {diff_sway:.1f}")
        c3.metric("③ スイングテンポ", f"{int(score_tempo)}点", f"時間差: {diff_time:.2f}秒")
        
        st.caption(f"プロのダウンスイング時間: {pro_down_time:.2f}秒 / あなた: {my_down_time:.2f}秒")

        # --- アドバイス ---
        with st.expander("💡 AIコーチからのアドバイス", expanded=True):
            if score_tempo < 80:
                if my_down_time > pro_down_time:
                    st.write("❌ **テンポ**: プロよりスイングがゆっくりです。思い切って振り抜きましょう！")
                else:
                    st.write("❌ **テンポ**: プロより速すぎます（打ち急ぎ）。トップで一瞬「間」を作ると安定します。")
            else:
                st.write("✅ **テンポ**: 素晴らしいリズムです！プロ並みのキレがあります。")

        # --- 同期動画生成 ---
        st.divider()
        st.subheader("🎬 フォーム比較 (同期再生)")
        if st.button("✨ 同期比較動画を生成する"):
            with st.spinner("生成中..."):
                tfile_sync = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                target_fps = min(st.session_state['pro_fps'], st.session_state['my_fps'])
                create_sync_video(
                    st.session_state['pro_processed_video'],
                    st.session_state['my_processed_video'],
                    pro_df.iloc[pro_top_idx]['Frame'],
                    my_df.iloc[my_top_idx]['Frame'],
                    tfile_sync.name,
                    target_fps
                )
                st.session_state['sync_video_path'] = tfile_sync.name
                st.success("生成完了")

        if st.session_state['sync_video_path']:
            st.video(st.session_state['sync_video_path'])
