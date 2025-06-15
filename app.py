from requests import session
import streamlit as st
import pandas as pd
import urllib.parse
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import hashlib

# ---------- 資料庫連線 ----------
def get_engine():
    return create_engine("mysql+mysqlconnector://root:877899@localhost/music_db_new")  # 根據你的環境調整

# ---------- 密碼雜湊 ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------- 使用者功能 ----------
def register_user(username, password):
    password_hash = hash_password(password)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM Users WHERE username = :username"), {"username": username})
        if result.fetchone():
            return False, "帳號已存在"
        conn.execute(text("INSERT INTO Users (username, password_hash) VALUES (:username, :password_hash)"),
                     {"username": username, "password_hash": password_hash})
        conn.commit()
        return True, "註冊成功！"

def login_user(username, password):
    password_hash = hash_password(password)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM Users WHERE username = :username AND password_hash = :password_hash"),
                              {"username": username, "password_hash": password_hash})
        return result.fetchone() is not None

# 刪除使用者（需密碼驗證）
def delete_user(username, password, current_user):
    if username != current_user:
        return False, "只能刪除自己的帳號"
    password_hash = hash_password(password)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM Users WHERE username = :username AND password_hash = :password_hash"),
            {"username": username, "password_hash": password_hash}
        )
        if result.fetchone() is None:
            return False, "帳號或密碼錯誤"
        conn.execute(
            text("DELETE FROM Users WHERE username = :username"),
            {"username": username}
        )
        conn.commit()
        return True, "帳號已成功刪除"

# 顯示所有使用者
def show_users():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT username FROM Users"))
        return [row[0] for row in result]

# ---------- 搜尋歌曲 ----------
def search_songs(keyword, genre, year_range):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT 
                s.title, s.artist, 
                GROUP_CONCAT(g.name ORDER BY g.name SEPARATOR ', ') AS genres,
                YEAR(s.release_date) AS year, s.emotion
            FROM Songs s
            JOIN Song_Genres sg ON s.song_id = sg.song_id
            JOIN Genres g ON sg.genre_id = g.genre_id
            WHERE (s.title LIKE %(kw)s OR s.artist LIKE %(kw)s)
              AND (%(genre)s = '全部' OR g.name = %(genre)s)
              AND (s.release_date IS NOT NULL AND YEAR(s.release_date) BETWEEN %(min_year)s AND %(max_year)s)
            GROUP BY s.song_id, s.title, s.artist, s.release_date, s.emotion
            LIMIT 50
        """, conn, params={
            "kw": f"%{keyword}%",
            "genre": genre,
            "min_year": year_range[0],
            "max_year": year_range[1]
        })
    if not df.empty:
        df["YouTube"] = df.apply(
            lambda row: f"<a href='https://www.youtube.com/results?search_query={'+'.join(row['title'].split())}+{'+'.join(row['artist'].split())}' target='_blank'>🔗</a>",
            axis=1
        )
    return df

# ---------- 分群推薦 ----------
def get_cluster_recommendations():
    engine = get_engine()
    with engine.connect() as conn:
        df_all = pd.read_sql("""
            SELECT song_id, title, artist, energy, danceability, positiveness, speechiness,
                   liveness, acousticness, instrumentalness
            FROM Songs WHERE release_date IS NOT NULL
        """, conn)

    features = ["energy", "danceability", "positiveness", "speechiness", "liveness", "acousticness", "instrumentalness"]
    df_all = df_all.dropna(subset=features)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_all[features])
    df_all["cluster"] = KMeans(n_clusters=10, random_state=42).fit_predict(X)

    sampled = df_all.groupby("cluster").apply(lambda g: g.sample(1, random_state=42)).reset_index(drop=True)
    sampled["YouTube"] = sampled.apply(
        lambda row: f"https://www.youtube.com/results?search_query={'+'.join(row['title'].split())}+{'+'.join(row['artist'].split())}",
        axis=1
    )
    return sampled

# ---------- 人格推論 ----------
def infer_personality(liked_ids):
    engine = get_engine()
    id_list = ",".join(str(i) for i in liked_ids)
    with engine.connect() as conn:
        avg_row = pd.read_sql(f"""
            SELECT AVG(energy) AS energy, AVG(danceability) AS danceability,
                   AVG(positiveness) AS positiveness, AVG(speechiness) AS speechiness,
                   AVG(liveness) AS liveness, AVG(acousticness) AS acousticness,
                   AVG(instrumentalness) AS instrumentalness
            FROM Songs WHERE song_id IN ({id_list})
        """, conn).iloc[0]
        df_types = pd.read_sql("SELECT * FROM Personality_Types", conn)

    def classify(val): return "low" if val < 40 else "mid" if val < 70 else "high"
    user_levels = {k: classify(avg_row[k]*100) for k in avg_row.index}
    def score(row): return sum(row[f"{f}_level"] == user_levels[f] for f in user_levels)
    df_types["match_score"] = df_types.apply(score, axis=1)
    df_types["match_percent"] = df_types["match_score"] / df_types["match_score"].sum()
    best = df_types.sort_values("match_score", ascending=False).iloc[0]
    return best["personality_type"], best["description"], df_types

# ---------- 圖表 ----------
def normalize_feature_series(avg_row, stats):
    return pd.Series({
        feature: 0.0 if pd.isna(avg_row[feature]) or pd.isna(stats[f"{feature}_max"]) else
        (avg_row[feature] - stats[f"{feature}_min"]) / (stats[f"{feature}_max"] - stats[f"{feature}_min"])
        if stats[f"{feature}_max"] != stats[f"{feature}_min"] else 0.0
        for feature in avg_row.index
    })

def plot_radar_chart_plotly(vector, title="你的音樂特徵雷達圖"):
    feature_map = {
        "energy": "活力", "danceability": "舞動性", "positiveness": "正向情緒",
        "speechiness": "語音成分", "liveness": "現場感", "acousticness": "原聲程度",
        "instrumentalness": "器樂性"
    }
    keys = list(feature_map.keys())
    vector = vector[keys].fillna(0).astype(float)
    labels = [feature_map[k] for k in keys] + [feature_map[keys[0]]]
    values = vector.tolist() + [vector[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself',
                                  line=dict(color='rgba(0,123,255,0.8)', width=3),
                                  fillcolor='rgba(0,123,255,0.2)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=False, title=title)
    st.plotly_chart(fig, use_container_width=True)

def plot_personality_match_bar(df_types):
    df_plot = df_types[["personality_type", "match_percent"]].sort_values("match_percent")
    fig = go.Figure(go.Bar(x=df_plot["match_percent"] * 100,
                           y=df_plot["personality_type"],
                           orientation="h",
                           text=[f"{p:.1%}" for p in df_plot["match_percent"]],
                           textposition="auto",
                           marker_color='rgba(0,123,255,0.7)'))
    fig.update_layout(title="人格型態相似度分析", xaxis_title="相似度 (%)", yaxis_title="人格型態", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 靈魂配對 ----------
def get_compatible_type(my_type, scenario):
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT compatible_type, description, match_score FROM Personality_Match
            WHERE personality_type = :my_type AND scenario = :scenario
            ORDER BY match_score DESC LIMIT 1
        """), {"my_type": my_type, "scenario": scenario}).fetchone()
        return result if result else (None, "尚無配對資料", None)

def fetch_matching_users(my_type):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT u.username, pr.personality_type FROM Users u
            JOIN personality_result pr ON u.user_id = pr.user_id
            WHERE pr.personality_type = %(my_type)s
        """, conn, params={"my_type": my_type})
        return df

# ---------- 配對歌單推薦 ----------
def fetch_common_style_songs(type1, type2):
    engine = get_engine()
    with engine.connect() as conn:
        # 先抓出各自的推薦清單
        df1 = pd.read_sql(text("""
            SELECT s.song_id, s.title, s.artist, s.energy, s.danceability, s.positiveness,
                   s.speechiness, s.liveness, s.acousticness, s.instrumentalness
            FROM Songs s
            JOIN Recommended_Songs r ON s.song_id = r.song_id
            WHERE r.personality_type = :type1
        """), conn, params={"type1": type1})

        df2 = pd.read_sql(text("""
            SELECT s.song_id, s.title, s.artist, s.energy, s.danceability, s.positiveness,
                   s.speechiness, s.liveness, s.acousticness, s.instrumentalness
            FROM Songs s
            JOIN Recommended_Songs r ON s.song_id = r.song_id
            WHERE r.personality_type = :type2
        """), conn, params={"type2": type2})

    # 找出交集
    df_common = df1[df1["song_id"].isin(df2["song_id"])]

    if not df_common.empty:
        df_common = df_common.sample(n=min(10, len(df_common)), random_state=42)
    else:
        # 若沒有交集，就找最相似的兩首（type1 的一首 vs type2 的一首）
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        feats = ["energy", "danceability", "positiveness", "speechiness", "liveness", "acousticness", "instrumentalness"]
        scaler = StandardScaler()
        df1_scaled = scaler.fit_transform(df1[feats])
        df2_scaled = scaler.transform(df2[feats])
        sim = cosine_similarity(df1_scaled, df2_scaled)

        idx1, idx2 = np.unravel_index(np.argmax(sim), sim.shape)
        df_common = pd.DataFrame([df1.iloc[idx1]])

    # 加入 YouTube 欄位
    df_common["YouTube"] = df_common.apply(
        lambda row: f"<a href='https://www.youtube.com/results?search_query={urllib.parse.quote_plus(row['title'] + ' ' + row['artist'])}' target='_blank'>🔗</a>",
        axis=1
    )
    return df_common

def display_soulmate_recommendations(type1, type2, match_score=None):
    st.subheader(f"為你與「{type2}」類型使用者推薦的靈魂共鳴歌單")
    df_common = fetch_common_style_songs(type1, type2)
    if df_common.empty:
        st.info("暫無共通推薦歌曲")
        return

    playlist_name = f"{type1} ♡ {type2} 靈魂共振歌單"
    if match_score is None:
        message = "你們的旋律共鳴指數突破天際，是音樂宇宙的雙生星座！"
    elif match_score > 0.9:
        message = "你們宛如命中注定的旋律夥伴，默契值爆表！"
    elif match_score > 0.8:
        message = "你們音樂頻率一致，是彼此最佳拍檔！"
    elif match_score > 0.7:
        message = "你們有著穩定的音樂共鳴，值得多多探索！"
    else:
        message = "音樂風格雖異，卻可能擦出驚喜火花！"

    st.markdown(f"##### 🎧 **{playlist_name}**")
    st.caption(message)

    df_display = df_common[["title", "artist", "YouTube"]].rename(columns={
        "title": "歌曲名稱", "artist": "演出者", "YouTube": "試聽連結"
    })
    st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

# ---------- 主畫面整合 ----------
def show_music_soulmate_section():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT personality_type FROM personality_result pr
            JOIN Users u ON pr.user_id = u.user_id
            WHERE u.username = :username
        """), {"username": st.session_state.username}).fetchone()

    if result:
        st.info(f"你的人格類型：**{result[0]}**")
    else:
        st.warning("⚠️ 你尚未進行人格測驗，請先前往『人格推薦』頁面完成測驗")
    st.header("🎼 音樂靈魂配對系統")
    my_type = st.selectbox("請選擇你的人格類型：", ["活力開朗型", "派對社交型", "語言思考型", "沉靜內斂型", "療癒放鬆型", "多元探索型"])
    scenario = st.selectbox("你想要配對的情境是：", ["讀書", "放空", "運動"])

    compatible_type, description, score = get_compatible_type(my_type, scenario)
    if compatible_type:
        st.success(f"與你最相容的類型是：**{compatible_type}**")
        st.caption(description)

        matching_users = fetch_matching_users(compatible_type)
        if not matching_users.empty:
            st.write(f"目前有 {len(matching_users)} 位 {compatible_type} 類型的使用者：")
            st.dataframe(matching_users, use_container_width=True)

        display_soulmate_recommendations(my_type, compatible_type, match_score=score)
    else:
        st.info("目前尚無配對建議。")
# ---------- 主程式 ----------
def main():
    st.set_page_config(page_title="音樂推薦與人格預測", layout="wide")
    st.title("🎵 音樂推薦與人格預測系統")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["🔐 登入", "📝 註冊"])

        with tab1:
            username = st.text_input("帳號", key="login_user")
            password = st.text_input("密碼", type="password", key="login_pass")
            if st.button("登入"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"歡迎回來，{username}！")
                else:
                    st.error("帳號或密碼錯誤")

        with tab2:
            new_user = st.text_input("新帳號", key="reg_user")
            new_pass = st.text_input("新密碼", type="password", key="reg_pass")
            if st.button("註冊"):
                if new_user and new_pass:
                    success, msg = register_user(new_user, new_pass)
                    if success:
                        st.success(msg)
                        st.write("請妥善保管您的帳號資訊，日後登入或刪除需用密碼驗證。")
                    else:
                        st.error(msg)
                else:
                    st.warning("請輸入帳號與密碼")
        return

    st.sidebar.markdown(f"👤 目前使用者：**{st.session_state.username}**")
    if st.sidebar.button("🚪 登出"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    page = st.sidebar.radio("功能選單", ["🔍 搜尋歌曲", "🎧 人格推薦", "💞 靈魂配對", "⚙️ 帳號管理"])

    if page == "🎧 人格推薦":
        st.header("🎧 勾選喜歡的歌曲以預測人格")
        df = get_cluster_recommendations()
        selected = []

        for _, row in df.iterrows():
            col1, col2 = st.columns([0.05, 0.95])
            if col1.checkbox("", key=row["song_id"]):
                selected.append(row["song_id"])
            col2.markdown(f"**{row['title']} - {row['artist']}** [🔗]({row['YouTube']})", unsafe_allow_html=True)

        if st.button("送出喜好"):
            if not selected:
                st.warning("請至少選一首歌")
            else:
                personality, desc, df_types = infer_personality(selected)
                st.success(f"你可能的人格型態是：**{personality}**")
                st.write(desc)

                engine = get_engine()
                with engine.begin() as conn:
                    user_id = conn.execute(
                        text("SELECT user_id FROM Users WHERE username = :username"),
                        {"username": st.session_state.username}
                    ).scalar()

                    for song_id in selected:
                        try:
                            conn.execute(text("""
                                INSERT INTO User_Selected_Songs (user_id, song_id)
                                VALUES (:user_id, :song_id)
                                ON DUPLICATE KEY UPDATE selected_at = CURRENT_TIMESTAMP
                            """), {"user_id": user_id, "song_id": song_id})
                        except Exception as e:
                            st.error(f"更新使用者選擇歌曲時發生錯誤: {e}")

                    avg_row = pd.read_sql(f"""
                        SELECT AVG(energy) AS energy, AVG(danceability) AS danceability,
                               AVG(positiveness) AS positiveness, AVG(speechiness) AS speechiness,
                               AVG(liveness) AS liveness, AVG(acousticness) AS acousticness,
                               AVG(instrumentalness) AS instrumentalness
                        FROM Songs WHERE song_id IN ({','.join(str(i) for i in selected)})
                    """, conn).iloc[0]

                    feature_map = {
                        "energy": "活力",
                        "danceability": "舞動性",
                        "positiveness": "正向情緒",
                        "speechiness": "語音成分",
                        "liveness": "現場感",
                        "acousticness": "原聲程度",
                        "instrumentalness": "器樂性"
                    }
                    avg_df = pd.DataFrame({
                        "特徵": [feature_map[k] for k in avg_row.index],
                        "平均值 (0~1)": avg_row.values
                    })
                    st.markdown("### 🔍 你的音樂特徵平均值")
                    st.dataframe(avg_df.style.format({"平均值 (0~1)": "{:.2f}"}))

                    try:
                        conn.execute(text("""
                            INSERT INTO personality_result 
                            (user_id, personality_type, avg_energy, avg_danceability, avg_speechiness, avg_acousticness, avg_positiveness, avg_liveness, avg_instrumentalness)
                            VALUES
                            (:user_id, :personality_type, :avg_energy, :avg_danceability, :avg_speechiness, :avg_acousticness, :avg_positiveness, :avg_liveness, :avg_instrumentalness)
                            ON DUPLICATE KEY UPDATE
                            personality_type = VALUES(personality_type),
                            avg_energy = VALUES(avg_energy),
                            avg_danceability = VALUES(avg_danceability),
                            avg_speechiness = VALUES(avg_speechiness),
                            avg_acousticness = VALUES(avg_acousticness),
                            avg_positiveness = VALUES(avg_positiveness),
                            avg_liveness = VALUES(avg_liveness),
                            avg_instrumentalness = VALUES(avg_instrumentalness)
                        """), {
                            "user_id": user_id,
                            "personality_type": personality,
                            "avg_energy": float(avg_row["energy"]),
                            "avg_danceability": float(avg_row["danceability"]),
                            "avg_speechiness": float(avg_row["speechiness"]),
                            "avg_acousticness": float(avg_row["acousticness"]),
                            "avg_positiveness": float(avg_row["positiveness"]),
                            "avg_liveness": float(avg_row["liveness"]),
                            "avg_instrumentalness": float(avg_row["instrumentalness"])
                        })
                    except Exception as e:
                        st.error(f"新增人格結果時發生錯誤: {e}")

                df_recommended = get_cluster_recommendations()
                stats = {f"{f}_max": df_recommended[f].max() for f in avg_row.index}
                stats.update({f"{f}_min": df_recommended[f].min() for f in avg_row.index})
                normalized = normalize_feature_series(avg_row, stats)

                plot_radar_chart_plotly(normalized)
                plot_personality_match_bar(df_types)

    elif page == "🔍 搜尋歌曲":
        st.header("🔍 音樂查詢系統")
        keyword = st.text_input("關鍵字（歌名或歌手）")

        with get_engine().connect() as conn:
            genre_options = ["全部"] + pd.read_sql("SELECT name FROM Genres ORDER BY name", conn)['name'].dropna().tolist()
        genre = st.selectbox("音樂類型", genre_options)
        year_range = st.slider("年代範圍", 1950, 2025, (2000, 2023))

        if st.button("搜尋"):
            if keyword.strip() == "":
                st.warning("請輸入關鍵字")
            else:
                results = search_songs(keyword, genre, year_range)
                if results.empty:
                    st.info("找不到符合條件的歌曲。")
                else:
                    st.markdown(results.to_html(escape=False, index=False), unsafe_allow_html=True)

    elif page == "⚙️ 帳號管理":
        st.header("⚙️ 帳號管理")

        tab1, tab2 = st.tabs(["🧾 查看所有帳號", "🗑️ 刪除帳號"])

        with tab1:
            users = show_users()
            if users:
                st.write("目前帳號列表：")
                st.write(", ".join(users))
            else:
                st.info("目前尚無使用者。")

        with tab2:
            st.warning("⚠️ 刪除帳號後將無法復原，請謹慎操作。")
            del_user = st.text_input("請輸入帳號", key="delete_user")
            del_pass = st.text_input("請輸入密碼", type="password", key="delete_pass")
            if st.button("刪除帳號"):
                success, msg = delete_user(del_user, del_pass, st.session_state.username)
                if success:
                    if del_user == st.session_state.username:
                        st.success(msg)
                        st.session_state.logged_in = False
                        st.session_state.username = None
                        st.rerun()
                    else:
                        st.success(msg)
                else:
                    st.error(msg)

    elif page == "💞 靈魂配對":
        show_music_soulmate_section()

if __name__ == "__main__":
    main()