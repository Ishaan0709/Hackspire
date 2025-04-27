import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set it in .env file.")
    st.stop()

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Load Doctor Profile
DOCTOR_PROFILE_PATH = "Dr_Kshitij_Bhatnagar_Advanced_Profile.pdf"

def load_doctor_profile(path):
    try:
        loader = PyMuPDFLoader(path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        db = FAISS.from_documents(chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"Error loading Doctor Profile: {e}")
        return None

def smart_symptom_response(query):
    query = query.lower()
    responses = []

    if "cough" in query or "cold" in query:
        responses.append("""
**🩺 Condition: Cough/Cold**

**Precautions:**
- Take steam inhalation twice daily
- Stay away from dust and allergens
- Keep yourself warm

**Medicines:**
- Levocetirizine 5mg at night for 5 days
- Paracetamol 500mg if fever present

**General Advice:**
- Drink 2-3 liters of water daily
- Consult doctor if symptoms worsen
""")

    if "fever" in query:
        responses.append("""
**🩺 Condition: Fever**

**Precautions:**
- Monitor your temperature regularly
- Stay hydrated

**Medicines:**
- Paracetamol 500mg every 6 hours if fever >100°F

**General Advice:**
- Consult doctor if fever persists beyond 48 hours
""")

    if "ear pain" in query:
        responses.append("""
**🩺 Condition: Ear Pain**

**Precautions:**
- Keep the ear dry
- Avoid inserting any objects into the ear

**Medicines:**
- Amoxicillin 500mg thrice daily for 5-7 days (if prescribed)

**General Advice:**
- ENT specialist consultation advisable if pain worsens
""")

    if "throat pain" in query:
        responses.append("""
**🩺 Condition: Throat Pain**

**Precautions:**
- Warm saline gargles 3 times a day
- Avoid spicy food and cold beverages

**Medicines:**
- Lozenges (e.g., Strepsils) as needed

**General Advice:**
- Visit a doctor if throat pain persists more than 3 days
""")

    return "\n\n".join(responses) if responses else None

def generate_summary(conversation, patient_name):
    summary = f"""
# 📄 Consultation Summary Report

**Patient Name:** {patient_name}  
**Total Questions Asked:** {len([msg for msg in conversation if msg[0] == "Patient"])}  
**Consultation Status:** {'Approved' if st.session_state.get('approval_status', False) else 'Pending Approval'}

## Conversation History:
"""
    for speaker, message in conversation:
        summary += f"\n**{speaker}:**\n{message}\n"

    summary += "\n---\n"
    summary += "**Note:** This is an AI-generated consultation summary."
    
    return summary

def main():
    st.set_page_config(page_title="Doctor AI Assistant", page_icon="🩺", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
        body, .stApp {
            color: white !important;
        }
        .welcome-box {
            background-color: #1e3a8a;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            color: white !important;
        }
        .patient-message {
            background: #2d4b8e;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 5px solid #2196f3;
            color: white !important;
        }
        .doctor-message {
            background: transparent;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 5px solid #2196f3;
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6, strong {
            color: white !important;
        }
        .st-b7, .st-b8, .st-b9 {
            color: white !important;
        }
        .stTextInput input {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🩺 Dr. Kshitij Bhatnagar - AI Consultation Assistant")

    # Initialize session state
    if "db" not in st.session_state:
        st.session_state.db = load_doctor_profile(DOCTOR_PROFILE_PATH)
    if not st.session_state.db:
        st.stop()

    defaults = {
        "conversation": [],
        "consultation_done": False,
        "waiting_for_approval": False,
        "approval_status": False,
        "patient_name": "",
        "initialized": True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    user_role = st.sidebar.selectbox("Select your role:", ("Patient", "Doctor"))

    if user_role == "Patient":
        if not st.session_state.patient_name:
            with st.container():
                st.markdown("""
                <div class="welcome-box">
                    <h3>👋 Welcome to AI Medical Consultation</h3>
                    <p>Get preliminary medical advice before doctor's review</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.patient_name = st.text_input("Your full name:")
                if st.session_state.patient_name:
                    st.rerun()
                return

        st.subheader(f"💬 Consultation for {st.session_state.patient_name}")
        
        if not st.session_state.consultation_done:
            with st.form(key='consultation_form'):
                user_query = st.text_area("Describe your symptoms:", 
                                       placeholder="Example: I've had fever and headache since yesterday...",
                                       height=150)
                
                if st.form_submit_button("Get AI Advice"):
                    if not user_query.strip():
                        st.warning("Please describe your symptoms!")
                    else:
                        with st.spinner("Analyzing your symptoms..."):
                            smart_response = smart_symptom_response(user_query)
                            
                            if not smart_response:
                                model = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.7)
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=model,
                                    retriever=st.session_state.db.as_retriever(),
                                    chain_type="stuff"
                                )
                                smart_response = qa_chain.run(user_query)

                            st.session_state.conversation.append(("Patient", user_query))
                            st.session_state.conversation.append(("Doctor AI", smart_response))
                            st.rerun()

            if st.session_state.conversation:
                st.subheader("📜 Conversation History")
                for speaker, message in st.session_state.conversation:
                    div_class = "patient-message" if speaker == "Patient" else "doctor-message"
                st.markdown("""<div class="{}"><strong>{}:</strong><br>{}</div>
                            """.format(div_class, speaker, message.replace('\n', '<br>')), 
                            unsafe_allow_html=True)

                
                st.markdown("---")
                if st.button("✅ End Consultation & Generate Summary", type="primary"):
                    st.session_state.consultation_done = True
                    st.session_state.waiting_for_approval = True
                    st.rerun()

        elif st.session_state.waiting_for_approval:
            st.subheader("🕒 Consultation Status")
            st.info("⌛ Waiting for Doctor's Approval... Please wait.")
            st.markdown(generate_summary(st.session_state.conversation, st.session_state.patient_name))

        elif st.session_state.consultation_done and not st.session_state.approval_status:
            st.subheader("❌ Consultation Status")
            st.error("Doctor has rejected this consultation and has appointed you for offline consulting.")

        elif st.session_state.approval_status:
            st.success("✅ Doctor has approved your consultation!")
            st.markdown(generate_summary(st.session_state.conversation, st.session_state.patient_name))
            
            if st.button("🔄 Start New Consultation"):
                for key in defaults:
                    if key != "db" and key != "initialized":
                        st.session_state[key] = defaults[key]
                st.rerun()

    elif user_role == "Doctor":
        st.subheader("👨‍⚕️ Doctor Approval Portal")
        
        if st.session_state.waiting_for_approval:
            st.info("ℹ️ New consultation awaiting review")
            st.markdown(generate_summary(st.session_state.conversation, st.session_state.patient_name))
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve Consultation", type="primary"):
                    st.session_state.approval_status = True
                    st.session_state.waiting_for_approval = False
                    st.success("Approved successfully!")
                    st.rerun()
            with col2:
                if st.button("❌ Request Changes"):
                    st.session_state.waiting_for_approval = False
                    st.session_state.consultation_done = True
                    st.session_state.approval_status = False
                    st.info("Patient has been advised offline consultation.")
                    st.rerun()
        else:
            st.info("No pending consultations at this time.")
            if st.session_state.approval_status:
                st.markdown("**Last approved consultation:**")
                st.markdown(generate_summary(st.session_state.conversation, st.session_state.patient_name))

if __name__ == "__main__":
    main()
