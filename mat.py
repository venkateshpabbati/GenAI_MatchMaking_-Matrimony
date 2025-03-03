import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"

# Load Google Gemini Model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# File paths
excel_file_path = "20250225_matrimony.xlsx"  # Path to your data file
male_index_file = "index_male.faiss"
female_index_file = "index_female.faiss"

# Streamlit UI
st.set_page_config(page_title="Matrimony Match Finder", layout="wide")
st.title("💍 AI-Powered Matrimony Match Finder")

# **Function to Create & Save FAISS Index**
def create_faiss_index(df, file_path):
    records = df.apply(lambda row: f"Name: {row['Name']}, Age: {row['Age']}, Marital Status: {row['Marital Status']}, "
                                   f"Education: {row['Education_Standardized']}, Profession: {row['Occupation']}, "
                                   f"Country: {row['Country']}, Height: {row['Hight/FT']}, "
                                   f"Salary: {row['Salary-PA']}, Denomination: {row['Denomination_dropdown']}, "
                                   f"AboutMe: {row['AboutMe']}, AboutFamily: {row['AboutFamily']}, "
                                   f"Prefering: {row['Prefering']}", axis=1).tolist()

    embeddings = np.array([embedding_model.embed_query(text) for text in records])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS Index
    faiss.write_index(index, file_path)

    return index, df

# **Function to Load Data & Create FAISS DBs**
def load_data():
    if not os.path.exists(excel_file_path):
        st.error(f"⚠️ Profile data not found! Ensure `{excel_file_path}` exists.")
        st.stop()

    df = pd.read_excel(excel_file_path)

    # Split Data into Male & Female Profiles
    df_male = df[df["gender"] == "Male"]
    df_female = df[df["gender"] == "Female"]

    # Load or Create FAISS Index for Males
    if os.path.exists(male_index_file):
        st.session_state.index_male = faiss.read_index(male_index_file)
        st.session_state.df_male = df_male
    else:
        st.session_state.index_male, st.session_state.df_male = create_faiss_index(df_male, male_index_file)

    # Load or Create FAISS Index for Females
    if os.path.exists(female_index_file):
        st.session_state.index_female = faiss.read_index(female_index_file)
        st.session_state.df_female = df_female
    else:
        st.session_state.index_female, st.session_state.df_female = create_faiss_index(df_female, female_index_file)

# **Load Data & FAISS DBs on Start**
load_data()

# **Match Criteria Inputs**
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("🚻 I am looking for a", ["Bride", "Groom"])
    age = st.slider("📅 Preferred Age", 18, 60, (25, 35))
    marital_status = st.selectbox("💍 Marital Status", ["Any", "Unmarried", "Divorced", "Widowed"])

with col2:
    education = st.selectbox("🎓 Preferred Education", ["Any", "High School", "Bachelors", "Masters", "PhD"])
    profession = st.selectbox("💼 Preferred Profession", ["Any", "Doctor", "Engineer", "Lawyer", "Teacher", "Business"])

with col3:
    country = st.selectbox("🌍 Preferred Country", ["Any", "India", "USA", "UK", "Canada", "Australia"])
    denomination = st.selectbox("🕌 Preferred Denomination", ["Any", "Muslim", "Hindu", "Christian", "Sikh", "Jewish"])

# **Find Match Button**
if st.button("🔍 Find My Match"):
    # Select the Correct FAISS DB
    target_gender = "Female" if gender == "Bride" else "Male"
    print(target_gender)
    index_file = male_index_file if target_gender == "Male" else female_index_file
    print(index_file)
    df_profiles = st.session_state.df_male if target_gender == "Male" else st.session_state.df_female
    print(df_profiles['gender'].unique())
    # Check if Index Exists
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        st.error("⚠️ No valid profile database found! Please check the data file.")
        st.stop()

    # Generate Query Based on Input
    query_text = f"Gender: {target_gender}, Age: {age}, Marital Status: {marital_status}, Education: {education}, "
    query_text += f"Profession: {profession}, Country: {country}, Denomination: {denomination}"
    query_vector = np.array([embedding_model.embed_query(query_text)])

    # Search in FAISS
    distances, indices = index.search(query_vector, k=10)  # Get top 10 matches
    st.subheader("💡 AI-Generated Best Matches:")

    matches = []
    for idx in indices[0]:
        if idx < len(df_profiles):
            match = df_profiles.iloc[idx]

            # Prepare Data for AI Query
            match_data = {
                "Name": match["Name"],
                "Email": match["email"],
                "Age": match["Age"],
                "Date of Birth": match["Date Of Birth"],
                "Marital Status": match["Marital Status"],
                "Denomination": match["Denomination_dropdown"],
                "City": match["City"],
                "Education": match["Education_Standardized"],
                "Occupation": match["Occupation"],
                "Salary (PA)": match["Salary-PA"],
                "Height": match["Hight/FT"],
                "AboutMe": match["AboutMe"],
                "AboutFamily": match["AboutFamily"],
                "Prefering": match["Prefering"]
            }

            matches.append(match_data)
            if len(matches) == 3:  # Show only top 3
                break

    if not matches:
        st.warning("❌ No suitable matches found. Try adjusting your preferences!")
    else:
        # AI-Powered Response Generation
        ai_prompt = f"""
        You are an AI matchmaking assistant. Based on the user's preference ({query_text}), generate a structured response with match details.
        
        **Format:**
        - **Match Name:**  
        - **About Me:**  
        - **About Family:**  
        - **Looking For:**  
        - **Profile Summary:** (Summarize age, marital status, education, occupation, Salary (PA), and other relevant details in a natural way)  

        Here are the matches:  
        {matches}
        """

        response = model.invoke(ai_prompt)
        st.write(response.content)
