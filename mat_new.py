import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ---------------------------
# 🔐 Secure API Key Handling
# ---------------------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("⚠️ Missing Google API Key! Please add it to `.streamlit/secrets.toml`")
    st.stop()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---------------------------
# 🤖 Load Gemini Models
# ---------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ---------------------------
# 📂 File Paths
# ---------------------------
EXCEL_FILE = "20250225_matrimony.xlsx"
INDEX_FILES = {"Male": "index_male.faiss", "Female": "index_female.faiss"}

# ---------------------------
# 🛠️ Helper: Create FAISS Index
# ---------------------------
def create_faiss_index(df: pd.DataFrame, file_path: str):
    records = df.apply(
        lambda row: f"Name: {row['Name']}, Age: {row['Age']}, Marital Status: {row['Marital Status']}, "
                    f"Education: {row['Education_Standardized']}, Profession: {row['Occupation']}, "
                    f"Country: {row['Country']}, Height: {row['Hight/FT']}, "
                    f"Salary: {row['Salary-PA']}, Denomination: {row['Denomination_dropdown']}, "
                    f"AboutMe: {row['AboutMe']}, AboutFamily: {row['AboutFamily']}, "
                    f"Prefering: {row['Prefering']}", axis=1
    ).tolist()

    embeddings = np.array([embedding_model.embed_query(text) for text in records])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, file_path)
    return index, df

# ---------------------------
# 🛠️ Load Data & Indexes
# ---------------------------
@st.cache_resource
def load_data():
    if not os.path.exists(EXCEL_FILE):
        st.error(f"⚠️ Profile data not found! Ensure `{EXCEL_FILE}` exists.")
        st.stop()

    df = pd.read_excel(EXCEL_FILE)
    session_data = {}

    for gender, file in INDEX_FILES.items():
        df_gender = df[df["gender"] == gender]
        if os.path.exists(file):
            session_data[f"index_{gender.lower()}"] = faiss.read_index(file)
        else:
            session_data[f"index_{gender.lower()}"], _ = create_faiss_index(df_gender, file)
        session_data[f"df_{gender.lower()}"] = df_gender

    return session_data

session_data = load_data()

# ---------------------------
# 🎨 Streamlit UI
# ---------------------------
st.set_page_config(page_title="💍 Matrimony Match Finder", layout="wide")
st.title("💍 AI-Powered Matrimony Match Finder")

col1, col2, col3 = st.columns(3)

with col1:
    looking_for = st.selectbox("🚻 I am looking for a", ["Bride", "Groom"])
    age_range = st.slider("📅 Preferred Age", 18, 60, (25, 35))
    marital_status = st.selectbox("💍 Marital Status", ["Any", "Unmarried", "Divorced", "Widowed"])

with col2:
    education = st.selectbox("🎓 Preferred Education", ["Any", "High School", "Bachelors", "Masters", "PhD"])
    profession = st.selectbox("💼 Preferred Profession", ["Any", "Doctor", "Engineer", "Lawyer", "Teacher", "Business"])

with col3:
    country = st.selectbox("🌍 Preferred Country", ["Any", "India", "USA", "UK", "Canada", "Australia"])
    denomination = st.selectbox("🕌 Preferred Denomination", ["Any", "Muslim", "Hindu", "Christian", "Sikh", "Jewish"])

# ---------------------------
# 🔍 Match Retrieval
# ---------------------------
if st.button("🔍 Find My Match"):
    target_gender = "Female" if looking_for == "Bride" else "Male"
    df_profiles = session_data[f"df_{target_gender.lower()}"]
    index = session_data[f"index_{target_gender.lower()}"]

    # ✅ Pre-filtering (structured constraints)
    df_filtered = df_profiles.copy()
    if marital_status != "Any":
        df_filtered = df_filtered[df_filtered["Marital Status"] == marital_status]
    if education != "Any":
        df_filtered = df_filtered[df_filtered["Education_Standardized"] == education]
    if profession != "Any":
        df_filtered = df_filtered[df_filtered["Occupation"] == profession]
    if country != "Any":
        df_filtered = df_filtered[df_filtered["Country"] == country]
    if denomination != "Any":
        df_filtered = df_filtered[df_filtered["Denomination_dropdown"] == denomination]
    df_filtered = df_filtered[(df_filtered["Age"] >= age_range[0]) & (df_filtered["Age"] <= age_range[1])]

    if df_filtered.empty:
        st.warning("❌ No profiles match your filters. Try adjusting preferences.")
        st.stop()

    # ✅ Vector Search on Filtered Profiles
    query_text = f"Looking for {target_gender} with preferences: Age {age_range}, Marital: {marital_status}, " \
                 f"Education: {education}, Profession: {profession}, Country: {country}, Denomination: {denomination}"
    query_vector = np.array([embedding_model.embed_query(query_text)])

    distances, indices = index.search(query_vector, k=20)
    matches = [df_filtered.iloc[i] for i in indices[0] if i < len(df_filtered)]

    # ✅ Re-ranking with Gemini
    rerank_prompt = f"""
    You are an expert AI matchmaker. Given the user's preferences:
    {query_text}
    
    Re-rank these matches (best first) and summarize each in max 3 sentences.
    {matches[:10]}
    """
    ranked_response = llm.invoke(rerank_prompt)

    # ---------------------------
    # 🎨 Display Results as Cards
    # ---------------------------
    st.subheader("💡 AI-Recommended Matches")

    for match in matches[:3]:  # Top 3
        with st.container():
            st.markdown(
                f"""
                <div style="border:1px solid #ddd; border-radius:15px; padding:15px; margin:10px; background:#f9f9f9;">
                    <h4>{match['Name']} ({match['Age']} yrs)</h4>
                    <p><b>📍 Location:</b> {match['City']}, {match['Country']}</p>
                    <p><b>💼 Profession:</b> {match['Occupation']}</p>
                    <p><b>🎓 Education:</b> {match['Education_Standardized']}</p>
                    <p><b>💰 Salary:</b> {match['Salary-PA']} PA</p>
                    <p><b>📝 About:</b> {match['AboutMe']}</p>
                    <p><b>👨‍👩‍👧 Family:</b> {match['AboutFamily']}</p>
                    <p><b>💘 Looking For:</b> {match['Prefering']}</p>
                </div>
                """, unsafe_allow_html=True
            )

    st.write("### 🧠 AI Reranked Summary")
    st.write(ranked_response.content)
