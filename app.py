import streamlit as st
import numpy as np
import joblib

# ---------------- Page Config ----------------
st.set_page_config(page_title="Mall Customer Clustering", layout="centered")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

.block-container {
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

h1 {
    text-align: center;
    font-weight: bold;
}

.stButton>button {
    background: #ff7a18;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background: #ff4b2b;
}

.result-box {
    background: rgba(0,0,0,0.3);
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("🛍️ Mall Customer Clustering")

# ---------------- Load Model ----------------
scaler = joblib.load("scalar.pkl")
model = joblib.load("kmeans.pkl")

st.markdown(f"**Model Clusters:** {model.n_clusters}")

# ---------------- Input Section ----------------
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income (k$)", min_value=0, value=50)

with col2:
    score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# ---------------- Cluster Information (0 to 9) ----------------
cluster_info = {
    0: {
        "type": "Low Income - Low Spending",
        "action": "Provide basic discounts and economical offers."
    },
    1: {
        "type": "High Income - High Spending (VIP Customers)",
        "action": "Provide premium services and exclusive deals."
    },
    2: {
        "type": "Low Income - High Spending",
        "action": "Offer budget-friendly promotions."
    },
    3: {
        "type": "High Income - Low Spending",
        "action": "Encourage spending through personalized marketing."
    },
    4: {
        "type": "Average Income - Average Spending",
        "action": "Maintain engagement with loyalty programs."
    },
    5: {
        "type": "Moderate Income - Selective Spending",
        "action": "Use targeted offers based on behavior."
    },
    6: {
        "type": "Young High Spenders",
        "action": "Promote trendy and premium products."
    },
    7: {
        "type": "Careful Customers",
        "action": "Provide discounts and value packs."
    },
    8: {
        "type": "Potential Customers",
        "action": "Engage with marketing campaigns and recommendations."
    },
    9: {
        "type": "Occasional Shoppers",
        "action": "Send seasonal offers and reminders."
    }
}

# ---------------- Prediction ----------------
if st.button("🔍 Predict Customer Segment"):
    
    data = np.array([[income, score]])
    data_scaled = scaler.transform(data)
    cluster = int(model.predict(data_scaled)[0])

    st.markdown(f"""
    <div class="result-box">
    <h3>Customer belongs to Cluster {cluster}</h3>
    </div>
    """, unsafe_allow_html=True)

    info = cluster_info.get(cluster, {
        "type": "New Customer Segment",
        "action": "Analyze customer behavior."
    })

    st.markdown(f"""
    <div class="result-box">
    <b>Customer Type:</b><br>
    {info['type']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-box">
    <b>Suggested Business Action:</b><br>
    {info['action']}
    </div>
    """, unsafe_allow_html=True)
