import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- 1. 基本設定 ---
st.set_page_config(layout="wide", page_title="K's Golf AI Coach Ultimate")

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
    }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #31333F; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State ---
if 'club_data' not in st.session_state: st.session_state['club_data'] = {}
if 'my_processed_video' not in st.session_state: st.session_state['my_processed_video'] = None
if 'my_df' not in st.session_state: st.session_state['my_df'] = None
if 'my_metrics' not in st.session_state: st.session_state['my_metrics'] = None

# --- 2. 計算・解析用関数 ---

def calculate_angle(a, b, c):
    """3点から角度を計算 (0-180度)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def get_vertical_angle(a, b):
    """2点（肩と腰など）と垂直線との角度（前傾角度用）"""
    a = np.array(a)
    b = np.array(b)
    # 垂直ベクトル
    v = np.array([b[0], a[1]]) 
    radians = np.arctan2(a[0]-b[0], a[1]-b[1])
    angle = np.abs(radians*180.0/np.pi)
    return angle

def analyze_video_advanced(input_path, output_path, rotate_mode="なし"):
    """
    高度な動画解析: 
    1. 骨格検知
    2. 各種メトリクス抽出 (腕、膝、前傾、頭)
    3. スイングイベント推定 (アドレス、トップ、インパクト)
    """
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 回転後のサイズ
    if rotate_mode in ["時計回りに90度", "反時計回りに90度"]:
        out_width, out_height = height, width
    else:
        out_width, out_height = width, height
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_data = []
    
    # メトリクス用リスト
    nose_x_list = []
    spine_angles = []
    knee_angles = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        bar = st.progress(0)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            
            # 回転処理
            if rotate_mode == "時計回りに90度":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_mode == "反時計回りに90度":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            frame_data = {
                "Frame": i,
                "Time": i / fps if fps > 0 else 0,
                "L_Wrist_Y": None,
                "Arm_Angle": None,
                "Spine_Angle": None,
                "R_Knee_Angle": None,
                "Nose_X": None
            }

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w, _ = image.shape
                
                # 必要な部位の座標 (正規化座標)
                nose = [lm[mp_pose.PoseLandmark.NOSE].x, lm[mp_pose.PoseLandmark.NOSE].y]
                l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]
                l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y]
                l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y]
                r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y]
                r_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y]
                r_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

                # 1. 左腕の角度 (トップでの伸び)
                arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                
                # 2. 前傾角度 (左肩と左腰を結ぶ線と垂直線の角度)
                spine_angle = get_vertical_angle(l_shoulder, l_hip)
                
                # 3. 右膝の角度 (スウェー/伸び上がりチェック)
                knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

                # データ格納
                frame_data["L_Wrist_Y"] = l_wrist[1]
                frame_data["Arm_Angle"] = arm_angle
                frame_data["Spine_Angle"] = spine_angle
                frame_data["R_Knee_Angle"] = knee_angle
                frame_data["Nose_X"] = nose[0]
                
                # リストに追加（統計用）
                nose_x_list.append(nose[0])
                spine_angles.append(spine_angle)
                knee_angles.append(knee_angle)

                # 描画 (スケルトン)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 頭の位置をマーキング
                cv2.circle(image, (int(nose[0]*w), int(nose[1]*h)), 8, (0, 255, 255), -1)

            pose_data.append(frame_data)
            out.write(image)
            if frame_count > 0: bar.progress((i + 1) / frame_count)

    cap.release()
    out.release()
    df = pd.DataFrame(pose_data)
    
    # --- スイングイベント推定 (簡易版) ---
    # Top: 手首(Y)が一番高い(値が小さい)フレーム
    if not df.empty and df['L_Wrist_Y'].notnull().any():
        top_idx = df['L_Wrist_Y'].idxmin()
        top_frame = df.loc[top_idx, 'Frame']
        
        # Address: 開始からTopまでの間で手首が一番低い位置 (簡易的)
        address_df = df[df['Frame'] < top_frame]
        address_frame = 0
        if not address_df.empty:
            address_frame = address_df['L_Wrist_Y'].idxmax() # 手が低い = Yが大きい
            # idxmaxだとindexが返るのでFrameを取得
            if pd.isna(address_frame): address_frame = 0
            else: address_frame = df.loc[address_frame, 'Frame']
        
        # Impact: Topの後で手首が一番低い位置 (ボール位置と仮定)
        impact_df = df[df['Frame'] > top_frame]
        impact_frame = frame_count - 1
        if not impact_df.empty:
            impact_idx = impact_df['L_Wrist_Y'].idxmax()
            impact_frame = df.loc[impact_idx, 'Frame']

        metrics = {
            'fps': fps,
            'top_frame': top_frame,
            'address_frame': address_frame,
            'impact_frame': impact_frame,
            'head_stability': np.std(nose_x_list) if nose_x_list else 0,
            'spine_stability': np.std(spine_angles) if spine_angles else 0,
            'knee_stability': np.std(knee_angles) if knee_angles else 0,
            'top_arm_angle': df.loc[top_idx, 'Arm_Angle']
        }
    else:
        metrics = None

    return output_path, df, metrics

def score_swing(pro_metrics, my_metrics):
    """スコア計算ロジック (5要素)"""
    scores = {}
    details = {}
    
    # 1. テンポ (Tempo) - 比率 3.0 が理想
    # Pro
    pro_backswing = pro_metrics['top_frame'] - pro_metrics['address_frame']
    pro_downswing = pro_metrics['impact_frame'] - pro_metrics['top_frame']
    pro_ratio = pro_backswing / pro_downswing if pro_downswing > 0 else 3.0
    
    # User
    my_backswing = my_metrics['top_frame'] - my_metrics['address_frame']
    my_downswing = my_metrics['impact_frame'] - my_metrics['top_frame']
    my_ratio = my_backswing / my_downswing if my_downswing > 0 else 0
    
    # 3.0からの乖離で採点
    diff_ratio = abs(3.0 - my_ratio)
    scores['Tempo'] = max(0, int(100 - diff_ratio * 30))
    details['Tempo'] = f"Ratio: {my_ratio:.2f} (Ideal: 3.0)"

    # 2. 左腕の伸び (Arm Extension)
    diff_arm = abs(pro_metrics['top_arm_angle'] - my_metrics['top_arm_angle'])
    scores['Arm'] = max(0, int(100 - diff_arm * 1.5))
    details['Arm'] = f"Angle: {my_metrics['top_arm_angle']:.1f}° (Pro: {pro_metrics['top_arm_angle']:.1f}°)"

    # 3. 頭の固定 (Head Stability) - 標準偏差の小ささ
    # ユーザーのブレが 0.03 (画面幅の3%) 以下なら満点に近い
    stab_score = max(0, int(100 - (my_metrics['head_stability'] * 1000))) 
    scores['Head'] = min(100, stab_score)
    details['Head'] = f"Stability: {my_metrics['head_stability']:.4f}"

    # 4. 前傾キープ (Spine) - 標準偏差
    spine_score = max(0, int(100 - (my_metrics['spine_stability'] * 50))) # 角度のブレ
    scores['Spine'] = min(100, spine_score)
    details['Spine'] = f"Variance: {my_metrics['spine_stability']:.2f}"

    # 5. 膝の固定 (Knee) - 標準偏差
    knee_score = max(0, int(100 - (my_metrics['knee_stability'] * 50)))
    scores['Knee'] = min(100, knee_score)
    details['Knee'] = f"Variance: {my_metrics['knee_stability']:.2f}"

    # 総合得点
    total = int(sum(scores.values()) / 5)
    return total, scores, details

# --- 3. サイドバー設定 ---
st.sidebar.title("⛳ Menu")
selected_club = st.sidebar.selectbox("使用クラブ", ["ドライバー", "フェアウェイウッド", "7番アイアン", "ウェッジ", "パター"])
app_mode = st.sidebar.radio("モード切替", ["1. プロ動画登録", "2. スイング解析 & スコア", "3. リアルタイム確認"])
st.sidebar.markdown("---")
st.sidebar.info(f"設定中: **{selected_club}**")

# --- 4. メインコンテンツ ---
st.title(f"🏌️ K's Golf AI Coach Ultimate")

# PAGE 1: プロ動画登録
if app_mode == "1. プロ動画登録":
    st.header(f"🧑‍🏫 {selected_club}のお手本設定")
    
    if selected_club in st.session_state['club_data']:
        st.success("✅ 設定済み")
        st.video(st.session_state['club_data'][selected_club]['video_path'])
        if st.button("リセット"):
            del st.session_state['club_data'][selected_club]
            st.rerun()
    else:
        pro_file = st.file_uploader("プロ動画をアップロード", type=['mp4', 'mov'])
        pro_rotate = st.selectbox("回転", ["なし", "時計回りに90度", "反時計回りに90度"])
        if pro_file and st.button("解析して保存"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(pro_file.read())
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            with st.spinner('AI解析中...'):
                processed_path, df, metrics = analyze_video_advanced(tfile.name, out_path, pro_rotate)
                if metrics:
                    st.session_state['club_data'][selected_club] = {'video_path': processed_path, 'metrics': metrics}
                    st.success("保存完了！")
                    st.rerun()
                else:
                    st.error("骨格が検出できませんでした。別の動画を試してください。")

# PAGE 2: ユーザー解析 & スコア
elif app_mode == "2. スイング解析 & スコア":
    st.header("📊 AI スイング診断")

    if selected_club not in st.session_state['club_data']:
        st.warning("まずは「プロ動画登録」でお手本を設定してください。")
    else:
        pro_data = st.session_state['club_data'][selected_club]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("お手本 (Pro)")
            st.video(pro_data['video_path'])
        with col2:
            st.subheader("あなた (You)")
            my_file = st.file_uploader("自分の動画", type=['mp4', 'mov'])
            my_rotate = st.selectbox("回転", ["なし", "時計回りに90度", "反時計回りに90度"])
            
            if my_file and st.button("診断開始"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(my_file.read())
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner('全集中で解析中...'):
                    processed_path, df, metrics = analyze_video_advanced(tfile.name, out_path, my_rotate)
                    st.session_state['my_processed_video'] = processed_path
                    st.session_state['my_metrics'] = metrics
                st.rerun()

            if st.session_state['my_processed_video']:
                st.video(st.session_state['my_processed_video'])

        # --- スコアカード表示 ---
        if st.session_state['my_metrics'] and pro_data['metrics']:
            total, scores, details = score_swing(pro_data['metrics'], st.session_state['my_metrics'])
            
            st.markdown("---")
            st.markdown(f"""
            <div class="score-card">
                <div>総合スコア</div>
                <div class="total-score">{total}</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            
            with c1:
                st.markdown('<div class="sub-score-box">⏱️ テンポ</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{scores["Tempo"]}</div>', unsafe_allow_html=True)
                st.caption(details['Tempo'])
            
            with c2:
                st.markdown('<div class="sub-score-box">💪 左腕の伸び</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{scores["Arm"]}</div>', unsafe_allow_html=True)
                st.caption(details['Arm'])
            
            with c3:
                st.markdown('<div class="sub-score-box">😐 頭の固定</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{scores["Head"]}</div>', unsafe_allow_html=True)
                st.caption(details['Head'])
                
            with c4:
                st.markdown('<div class="sub-score-box">📐 前傾維持</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{scores["Spine"]}</div>', unsafe_allow_html=True)
                st.caption(details['Spine'])
                
            with c5:
                st.markdown('<div class="sub-score-box">🦵 膝の粘り</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{scores["Knee"]}</div>', unsafe_allow_html=True)
                st.caption(details['Knee'])
                
            # アドバイス
            st.markdown("### 💡 AI Coach Advice")
            lowest_metric = min(scores, key=scores.get)
            if lowest_metric == "Tempo":
                st.warning("スイングのリズムが早すぎる、または遅すぎます。「イチ、ニ、サーン」のリズム（3:1）を意識しましょう。")
            elif lowest_metric == "Arm":
                st.warning("トップで左肘が曲がっています。遠くに上げるイメージで、アーク（円）を大きくしましょう。")
            elif lowest_metric == "Head":
                st.warning("頭が動きすぎています。ボールを最後まで見つめ、軸をブラさないようにしましょう。")
            elif lowest_metric == "Spine":
                st.warning("前傾姿勢が崩れています（起き上がり）。お尻の位置を変えない意識を持ちましょう。")
            elif lowest_metric == "Knee":
                st.warning("下半身が不安定です。右膝の角度をキープして、パワーを逃さないようにしましょう。")

# PAGE 3: リアルタイム
elif app_mode == "3. リアルタイム確認":
    st.header("🪞 リアルタイム・チェック")
    webrtc_streamer(key="realtime", mode=WebRtcMode.SENDRECV, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
