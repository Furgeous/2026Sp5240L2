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
        model="Salesforce/blip-image-captioning-large"
    )
    caption = captioner(image_path)[0]["generated_text"]
    return caption


def text2story(caption: str) -> str:
    story_generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    # Chat 格式，用 system + user 来精准控制风格
    messages = [
        {
            "role": "system",
            "content": "You are a creative storyteller for children aged 3-10. Write simple, fun, and imaginative stories."
        },
        {
            "role": "user",
            "content": f"Write a short children's story (50-100 words) based on this scene: {caption}"
        }
    ]
    # 使用 apply_chat_template 格式化输入
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    result = story_generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.3
    )
    raw = result[0]["generated_text"]
    # 截取 assistant 回复部分
    story = raw.split("<|assistant|>")[-1].strip()
    # 在最后一个句号截断
    last_period = story.rfind(".")
    if last_period != -1:
        story = story[:last_period + 1]
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
