# Program title: Storytelling App for Kids 🎨
# app.py

import streamlit as st
from transformers import pipeline
from gtts import gTTS
import os
import tempfile

# ── Functions ──────────────────────────────────────────────

def img2text(image_path: str) -> str:
    """Convert an uploaded image to a descriptive caption."""
    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )
    caption = captioner(image_path)[0]["generated_text"]
    return caption


def text2story(caption: str) -> str:
    """Expand a caption into a short children's story (50-100 words)."""
    story_generator = pipeline(
        "text-generation",
        model="roneneldan/TinyStories-33M"
    )
    prompt = f"Once upon a time, {caption}. "
    result = story_generator(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.3,
        num_return_sequences=1
    )
    story = result[0]["generated_text"]
    return story


def text2audio(story_text: str) -> str:
    """Convert story text to an MP3 audio file. Returns file path."""
    tts = gTTS(text=story_text, lang="en", slow=False)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name


# ── Main UI ────────────────────────────────────────────────

st.set_page_config(page_title="📖 Kids Story Maker", page_icon="🦜")
st.title("🦜 Turn Your Picture into a Story!")
st.markdown("*Upload a picture and listen to a magical story!* ✨")

uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save locally
    image_path = uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.image(uploaded_file, caption="Your Image", use_column_width=True)

    # Stage 1: Image → Caption
    with st.spinner("🔍 Looking at your picture..."):
        caption = img2text(image_path)
    st.success(f"🖼️ **What I see:** {caption}")

    # Stage 2: Caption → Story
    with st.spinner("✍️ Writing your story..."):
        story = text2story(caption)
    st.info(f"📖 **Your Story:**\n\n{story}")

    # Stage 3: Story → Audio
    with st.spinner("🎙️ Recording the story..."):
        audio_path = text2audio(story)

    st.audio(audio_path, format="audio/mp3")
    st.balloons()

    # Cleanup temp file
    os.remove(image_path)
