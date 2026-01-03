import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# --- 1. 基本設定 ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach")

st.markdown("""
    <style>
    .main > div {padding-top: 2rem;}
    video { width: 100% !important; height: auto !important; }
    
    .score-card {
        background-color: #262730;
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
        text-align: center;
    }
    .total-score { font-size: 3rem; font-weight: bold; color: #ff4b4b; }
    .sub-score-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        text-align: center;
        height: 100%;
    }
    .metric-value { font-size: 1.4rem; font-weight: bold; color: #31333F; }
    .advice-text { font-size: 0.9rem; color: #d32f2f; margin-top: 5px; font-weight: bold;}
    
    /* 安全警告（赤） */
    .safety-warning {
        background-color: #ffebee;
        color: #c62828;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ef9a9a;
        margin-bottom: 15px;
        font-weight: bold;
    }
    /* アングル案内（青） */
    .angle-info {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #90caf9;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State ---
# 構造: club_data[club][angle_type] = { ... }
if 'club_data' not in st.session_state: st.session_state['club_data'] = {}
if 'my_processed_video' not in st.session_state: st.session_state['my_processed_video'] = None
if 'my_df' not in st.session_state: st.session_state['my_df'] = None
if 'my_metrics' not in st.session_state: st.session_state['my_metrics'] = None
if 'sync_video_path' not in st.session_state: st.session_state['sync_video_path'] = None

# --- 2. 計算・解析用関数 ---

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_vertical_angle(a, b):
    a = np.array(a)
    b = np.array(b)
    radians = np.arctan2(abs(a[0]-b[0]), abs(a[1]-b[1]))
    angle = np.abs(radians*180.0/np.pi)
    return angle

def analyze_video_advanced(input_path, output_path, rotate_mode="なし"):
    """動画解析: 骨格検知とメトリクス抽出"""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if rotate_mode in ["時計回りに90度", "反時計回りに90度"]:
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_data = []
    
    nose_x_list = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        bar = st.progress(0)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            
            if rotate_mode == "時計回りに90度": frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_mode == "反時計回りに90度": frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            frame_data = {"Frame": i, "Time": i/fps if fps>0 else 0, "L_Wrist_Y": None}

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                nose = [lm[mp_pose.PoseLandmark.NOSE].x, lm[mp_pose.PoseLandmark.NOSE].y]
                l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]
                l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                spine_angle = get_vertical_angle(l_shoulder, l_hip)
                knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

                frame_data.update({
                    "L_Wrist_Y": l_wrist[1],
                    "Arm_Angle": arm_angle,
                    "Spine_Angle": spine_angle,
                    "R_Knee_Angle": knee_angle,
                    "Nose_X": nose[0]
                })
                nose_x_list.append(nose[0])
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            pose_data.append(frame_data)
            out.write(image)
            if frame_count > 0: bar.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    df = pd.DataFrame(pose_data)
    
    if not df.empty and df['L_Wrist_Y'].notnull().any():
        top_idx = df['L_Wrist_Y'].idxmin()
        top_frame = df.loc[top_idx, 'Frame']
        addr_df = df[df['Frame'] < top_frame]
        address_frame = df.loc[addr_df['L_Wrist_Y'].idxmax(), 'Frame'] if not addr_df.empty else 0
        imp_df = df[df['Frame'] > top_frame]
        impact_frame = df.loc[imp_df['L_Wrist_Y'].idxmax(), 'Frame'] if not imp_df.empty else frame_count-1
        
        top_data = df.loc[top_idx]
        metrics = {
            'fps': fps,
            'top_frame': int(top_frame),
            'address_frame': int(address_frame),
            'impact_frame': int(impact_frame),
            'head_stability': np.std(nose_x_list) if nose_x_list else 0,
            'spine_angle_top': top_data['Spine_Angle'],
            'knee_angle_top': top_data['R_Knee_Angle'],
            'top_arm_angle': top_data['Arm_Angle']
        }
    else:
        metrics = None

    return output_path, df, metrics

def create_sync_video(pro_path, my_path, pro_metrics, my_metrics, output_path):
    """同期動画生成"""
    cap_pro = cv2.VideoCapture(pro_path)
    cap_my = cv2.VideoCapture(my_path)
    h_pro = int(cap_pro.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_my = int(cap_my.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if h_pro == 0 or h_my == 0: return

    target_h = min(h_pro, h_my)
    w_pro = int(cap_pro.get(cv2.CAP_PROP_FRAME_WIDTH))
    w_my = int(cap_my.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_w_pro = int(w_pro * (target_h / h_pro))
    new_w_my = int(w_my * (target_h / h_my))
    target_w = new_w_pro + new_w_my

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, my_metrics['fps'], (target_w, target_h))

    pro_top = pro_metrics['top_frame']
    my_top = my_metrics['top_frame']
    
    offset = my_top - pro_top
    pro_delay = max(0, offset)
    my_delay = max(0, -offset)
    
    max_frames = int(max(cap_pro.get(cv2.CAP_PROP_FRAME_COUNT) + pro_delay, 
                         cap_my.get(cv2.CAP_PROP_FRAME_COUNT) + my_delay))

    bar = st.progress(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i in range(max_frames):
        if i < pro_delay:
            cap_pro.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_pro, frame_pro = cap_pro.read()
        else:
            ret_pro, frame_pro = cap_pro.read()
        
        if i < my_delay:
            cap_my.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_my, frame_my = cap_my.read()
        else:
            ret_my, frame_my = cap_my.read()
            
        if not ret_pro or not ret_my: break

        frame_pro_resized = cv2.resize(frame_pro, (new_w_pro, target_h))
        frame_my_resized = cv2.resize(frame_my, (new_w_my, target_h))
        concat_frame = cv2.hconcat([frame_pro_resized, frame_my_resized])
        
        sync_text = ""
        if i == (pro_top + pro_delay): sync_text = "TOP MATCH!"
        
        if sync_text:
            cv2.putText(concat_frame, sync_text, (target_w//2 - 150, 100), font, 1.5, (0,0,255), 5)
            cv2.putText(concat_frame, sync_text, (target_w//2 - 150, 100), font, 1.5, (255,255,255), 2)
        
        out.write(concat_frame)
        bar.progress((i+1)/max_frames)

    bar.progress(1.0)
    cap_pro.release()
    cap_my.release()
    out.release()
    return

def generate_advice(label, pro_val, my_val):
    diff = my_val - pro_val
    msg = ""
    score = 100
    abs_diff = abs(diff)
    if abs_diff < 5: score = 100
    else: score = max(0, int(100 - abs_diff * 2))

    if label == "Arm":
        if diff < -15: msg = f"⚠️ プロより{abs(int(diff))}°曲がっています。左腕をピンと伸ばして！"
        elif diff > 10: msg = "⚠️ 伸びすぎてロックしています。少しリラックス。"
        else: msg = "✅ Good! 綺麗に伸びています。"
    elif label == "Spine":
        if diff < -10: msg = f"⚠️ プロより{abs(int(diff))}°起きています。前傾キープ！"
        elif diff > 10: msg = f"⚠️ プロより{abs(int(diff))}°深く曲げすぎています。"
        else: msg = "✅ Good! 前傾姿勢が完璧です。"
    elif label == "Knee":
        if diff > 10: msg = f"⚠️ プロより{abs(int(diff))}°伸びて棒立ちです。"
        elif diff < -10: msg = "⚠️ 膝を曲げすぎています。"
        else: msg = "✅ Good! 膝が安定しています。"
    elif label == "Tempo":
        if my_val < 2.5: msg = "⚠️ 打ち急ぎです。バックスイングをゆったり。"
        elif my_val > 3.5: msg = "⚠️ 始動が遅すぎます。リズムよく！"
        else: msg = "✅ 完璧なリズム（3:1）です！"
        score = max(0, int(100 - abs(3.0 - my_val)*30))
    elif label == "Head":
        if my_val > pro_val * 2: msg = "⚠️ 頭が動きすぎています。軸を固定！"
        else: msg = "✅ Good! 体幹が強く安定しています。"
        score = max(0, int(100 - (my_val * 1000)))
    return score, msg

# --- 3. リアルタイム分析クラス ---
class RealtimeCoach(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.target_metrics = None 

    def update_target(self, metrics):
        self.target_metrics = metrics

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        cv2.putText(img, "AI Coach Eye", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            l_shoulder = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_elbow = [lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            l_wrist = [lm[self.mp_pose.PoseLandmark.LEFT_WRIST].x, lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            current_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            
            if self.target_metrics:
                target_arm = self.target_metrics['top_arm_angle']
                
                cv2.rectangle(img, (10, 60), (350, 180), (0,0,0), -1)
                
                cv2.putText(img, f"Current Arm: {int(current_arm_angle)} deg", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, f"Target (Pro): {int(target_arm)} deg", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                diff = current_arm_angle - target_arm
                if abs(diff) < 15:
                    cv2.putText(img, "GOOD POSE!", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                elif diff < -15:
                    cv2.putText(img, "Extend Arm!", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(img, "Relax Arm!", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "No Pro Data Selected", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return img

# --- 4. サイドバー設定 ---
st.sidebar.title("⛳ Menu")
selected_club = st.sidebar.selectbox("使用クラブ", ["ドライバー", "フェアウェイウッド", "7番アイアン", "ウェッジ", "パター"])
app_mode = st.sidebar.radio("モード切替", ["1. プロ動画登録", "2. スイング解析 & スコア", "3. 比較動画作成(Sync)", "4. リアルタイム・コーチ"])

st.sidebar.markdown("---")
st.sidebar.info(f"設定中: **{selected_club}**")

# --- 5. メインコンテンツ ---
st.title(f"🏌️ K's Golf AI Coach Professional")

# PAGE 1: プロ動画登録
if app_mode == "1. プロ動画登録":
    st.header(f"🧑‍🏫 {selected_club}のお手本設定")
    st.write("クラブごとに「後方」と「体の正面」の2種類を保存できます。")
    
    st.markdown("""
    <div class="safety-warning">
        ⚠️ 安全警告：打球の進行方向（ボールの飛び出す方向）には絶対に立たないでください。
        カメラは安全な距離を保って設置してください。
    </div>
    """, unsafe_allow_html=True)
    
    if selected_club not in st.session_state['club_data']:
        st.session_state['club_data'][selected_club] = {}

    # タブ名変更：後方をメインに
    tab_side, tab_front = st.tabs(["後方 (Down-the-line)", "体の正面 (Face-on)"])
    
    def register_pro_video(angle_key, angle_name):
        current_data = st.session_state['club_data'][selected_club].get(angle_key)
        if current_data:
            st.success(f"✅ {angle_name}動画: 設定済み")
            st.video(current_data['video_path'])
            if st.button(f"{angle_name}動画を削除", key=f"del_{angle_key}"):
                del st.session_state['club_data'][selected_club][angle_key]
                st.rerun()
        else:
            pro_file = st.file_uploader(f"プロの{angle_name}動画をアップロード", type=['mp4', 'mov'], key=f"up_{angle_key}")
            pro_rotate = st.selectbox("回転", ["なし", "時計回りに90度", "反時計回りに90度"], key=f"rot_{angle_key}")
            if pro_file and st.button(f"解析して保存 ({angle_name})", key=f"btn_{angle_key}"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(pro_file.read())
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('AIがスイングを解析中...'):
                    processed_path, df, metrics = analyze_video_advanced(tfile.name, out_path, pro_rotate)
                    if metrics:
                        st.session_state['club_data'][selected_club][angle_key] = {'video_path': processed_path, 'metrics': metrics}
                        st.success(f"{angle_name}データを保存しました！")
                        st.rerun()

    # タブの中身
    with tab_side:
        st.info("飛球線後方（背中側）から、ターゲット方向に向かって撮影した動画です。")
        register_pro_video('Side', '後方')
    with tab_front:
        st.info("体の正面（お腹側）から、体と直角になる位置で撮影した動画です。※打球方向に立たないこと！")
        register_pro_video('Front', '体の正面')

# PAGE 2: ユーザー解析 & スコア
elif app_mode == "2. スイング解析 & スコア":
    st.header("📊 AI スイング診断")

    if selected_club not in st.session_state['club_data'] or not st.session_state['club_data'][selected_club]:
        st.warning("まずは「プロ動画登録」でお手本を設定してください。")
    else:
        # アングル選択（UI表示を変更）
        available_angles = list(st.session_state['club_data'][selected_club].keys())
        # ラジオボタンの表示名を変換
        target_angle = st.radio(
            "どのアングルと比較しますか？", 
            available_angles, 
            format_func=lambda x: "体の正面 (Face-on)" if x=="Front" else "後方 (Down-the-line)"
        )
        
        pro_data = st.session_state['club_data'][selected_club][target_angle]
        pm = pro_data['metrics']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"お手本 ({'体の正面' if target_angle=='Front' else '後方'})")
            st.video(pro_data['video_path'])
        with col2:
            st.subheader("あなた (You)")
            
            warning_msg = "体の正面（お腹側）" if target_angle == "Front" else "後方（背中側・飛球線後方）"
            st.markdown(f"""
            <div class="safety-warning">
                ⚠️ <strong>撮影アングル注意:</strong><br>
                必ずプロと同じ <strong>「{warning_msg}」</strong> から撮影してください。<br>
                ※ 打球の進行方向には絶対に立たないでください。
            </div>
            """, unsafe_allow_html=True)

            my_file = st.file_uploader("自分の動画", type=['mp4', 'mov'])
            my_rotate = st.selectbox("回転", ["なし", "時計回りに90度", "反時計回りに90度"])
            
            if my_file and st.button("診断開始"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(my_file.read())
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('現在解析中... AIがスイングを分析しています'):
                    processed_path, df, metrics = analyze_video_advanced(tfile.name, out_path, my_rotate)
                    st.session_state['my_processed_video'] = processed_path
                    st.session_state['my_metrics'] = metrics
                st.rerun()

            if st.session_state['my_processed_video']:
                st.video(st.session_state['my_processed_video'])

        if st.session_state['my_metrics']:
            mm = st.session_state['my_metrics']
            m_back = mm['top_frame'] - mm['address_frame']
            m_down = mm['impact_frame'] - mm['top_frame']
            my_tempo = m_back / m_down if m_down > 0 else 0
            
            s_arm, m_arm = generate_advice("Arm", pm['top_arm_angle'], mm['top_arm_angle'])
            s_spine, m_spine = generate_advice("Spine", pm['spine_angle_top'], mm['spine_angle_top'])
            s_knee, m_knee = generate_advice("Knee", pm['knee_angle_top'], mm['knee_angle_top'])
            s_tempo, m_tempo = generate_advice("Tempo", 3.0, my_tempo)
            s_head, m_head = generate_advice("Head", pm['head_stability'], mm['head_stability'])

            total_score = int((s_arm + s_spine + s_knee + s_tempo + s_head) / 5)

            st.markdown("---")
            st.markdown(f"""
            <div class="score-card">
                <div>総合スコア</div>
                <div class="total-score">{total_score}</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            def show_card(col, title, score, msg):
                with col:
                    st.markdown(f'<div class="sub-score-box"><div>{title}</div><div class="metric-value">{score}</div><div class="advice-text">{msg}</div></div>', unsafe_allow_html=True)

            show_card(c1, "⏱️ テンポ", s_tempo, m_tempo)
            show_card(c2, "💪 左腕", s_arm, m_arm)
            show_card(c3, "😐 頭固定", s_head, m_head)
            show_card(c4, "📐 前傾", s_spine, m_spine)
            show_card(c5, "🦵 膝", s_knee, m_knee)

# PAGE 3: 比較動画 (Sync)
elif app_mode == "3. 比較動画作成(Sync)":
    st.header("🎞️ 同期動画作成")
    
    if selected_club in st.session_state['club_data'] and st.session_state['my_metrics']:
        available_angles = list(st.session_state['club_data'][selected_club].keys())
        target_angle = st.radio(
            "どのアングルのプロ動画と結合しますか？", 
            available_angles, 
            format_func=lambda x: "体の正面 (Face-on)" if x=="Front" else "後方 (Down-the-line)"
        )
        
        if st.button("比較動画を作成"):
            sync_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            with st.spinner("現在処理中... 動画を結合しています"):
                pro_path = st.session_state['club_data'][selected_club][target_angle]['video_path']
                my_path = st.session_state['my_processed_video']
                
                create_sync_video(
                    pro_path, 
                    my_path, 
                    st.session_state['club_data'][selected_club][target_angle]['metrics'],
                    st.session_state['my_metrics'],
                    sync_out
                )
                st.session_state['sync_video_path'] = sync_out
            st.success("完成しました！")
            
        if st.session_state['sync_video_path']:
            st.video(st.session_state['sync_video_path'])
    else:
        st.warning("まずは「プロ動画登録」と「スイング解析」を行ってください。")

# PAGE 4: リアルタイム・コーチ
elif app_mode == "4. リアルタイム・コーチ":
    st.header("📢 リアルタイム・AIコーチ")
    st.write("カメラに向かって構えてください。プロの数値と比較して、撮影者にアドバイスを表示します。")

    st.markdown("""
    <div class="safety-warning">
        ⚠️ 安全警告：撮影者は打球の進行方向には絶対に立たないでください。
        プレイヤーと十分な距離をとって撮影してください。
    </div>
    """, unsafe_allow_html=True)

    if selected_club not in st.session_state['club_data'] or not st.session_state['club_data'][selected_club]:
         st.warning("プロ動画が登録されていません。")
    else:
        available_angles = list(st.session_state['club_data'][selected_club].keys())
        target_angle = st.radio(
            "どのアングルでチェックしますか？", 
            available_angles, 
            format_func=lambda x: "体の正面 (Face-on)" if x=="Front" else "後方 (Down-the-line)"
        )
        
        target_metrics = st.session_state['club_data'][selected_club][target_angle]['metrics']
        
        ctx = webrtc_streamer(
            key="realtime-coach", 
            mode=WebRtcMode.SENDRECV, 
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=RealtimeCoach,
            async_processing=True
        )
        
        if ctx.video_processor:
            ctx.video_processor.update_target(target_metrics)
