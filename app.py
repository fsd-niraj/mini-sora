import streamlit as st
from main import generate_video_from_prompt
import os

st.set_page_config(page_title="Mini Sora", page_icon="🎥", layout="centered")

st.title("🎥 Mini Sora — Text to Video Playground")
st.write("Type a prompt below and I’ll generate a tiny synthetic video!")

# Chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "video" in msg:
            st.video(msg["video"])
        else:
            st.write(msg["content"])

# User input
prompt = st.chat_input("Describe the video you want...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate video
    with st.chat_message("assistant"):
        with st.spinner("🎬 Generating video..."):
            video_path = generate_video_from_prompt(prompt, duration=3, fps=12)
            if not os.path.exists(video_path):
                st.error("❌ Video generation failed. Please try again.")
            else:
              try:
                  st.video(video_path)
              except Exception as e:
                  st.error(f"❌ An error occurred while displaying the video: {e}")

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Here's your video!",
            "video": video_path
        })
