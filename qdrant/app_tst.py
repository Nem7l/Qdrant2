import base64
import math
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from transformers import AutoImageProcessor, ResNetForImageClassification
import streamlit as st
import os
collection_name = "instagram_data" 
header_image_url = "https://i.imgur.com/K9a89wH.jpeg"

# Display the header image
st.image(header_image_url, use_column_width=True)
   
if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None

 
 
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

def resize_image(pil_img, target_width=256, keep_aspect_ratio=True):
    original_width, original_height = pil_img.size
    image_aspect_ratio = original_width / original_height

    if keep_aspect_ratio:
        target_height = math.floor(target_width / image_aspect_ratio)
    else:
        target_height = original_height  # If aspect ratio is not maintained, use original height

    resized_img = pil_img.resize((target_width, target_height), Image.LANCZOS)  # Use LANCZOS for high-quality resizing
    return resized_img

def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="png")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def select_record(new_record):
    st.session_state.selected_record = new_record

def get_init_records():
    client = get_client()
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=6,
    )
    return records

def get_image_recommendation(encoded_img):
    image = Image.open(BytesIO(encoded_img))
    image = image.convert("RGB")
    image = resize_image(image)
    print(image.size)
    inputs = processor(images=image, return_tensors="pt")
    embed = model(**inputs)
    return embed.logits, convert_image_to_base64(image), image

def get_similar_records():
    client = get_client()
    if st.session_state.selected_record is None:
        return []

    record = st.session_state.selected_record
    records, _ = client.recommend(
        collection_name=collection_name,
        positive=[record.id],
        limit=6,
    )
    return records

def get_bytes_from_base64(base64_str):
    return BytesIO(base64.b64decode(base64_str))

@st.cache_resource
def get_client():
    return QdrantClient(
        url="https://a7d15b9f-649d-4c0f-9433-2a6c06cc66cc.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key="mQ3vUPBNtp0mRY6h0gTFMRncal8o1o_trEhck6tYv327wNB7c8WLHw",
    )

def search_similar_images(image_bytes, embed, encoded_img, orig_image):
    client = get_client()
    records = client.search(
        collection_name=collection_name,
        query_vector=embed[0].tolist(),
        with_vectors=True,
        with_payload=True,
        limit=8,
    )
    return records, encoded_img, orig_image


uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

num_similar_images = 2

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    embed, encoded_img, orig_image = get_image_recommendation(image_bytes)
    st.image(BytesIO(image_bytes), use_column_width=True)

    st.markdown("---")
    st.markdown("Here's Your Fellas ✨ !")

    records, encoded_img, orig_image = search_similar_images(image_bytes, embed, encoded_img, orig_image)

    row_items = 4
    rows = [records[i:i + row_items] for i in range(0, len(records), row_items)]

    for row in rows:
        columns = st.columns(row_items)
        for idx, record in enumerate(row):
            image_bytes = get_bytes_from_base64(record.payload['encoded_image'])
            with columns[idx]:
                st.image(image_bytes, use_column_width=True)
                st.markdown(f"<div style='display: flex; align-items: center;'><i class='fab fa-instagram' style='margin-right: 5px;'></i><a href='https://www.instagram.com/{record.payload['instagram_user']}' target='_blank' style='color: #a19eae; text-decoration: none;'>@{record.payload['instagram_user']}</a></div>", unsafe_allow_html=True)
                st.markdown(f"<a href='{record.payload['link']}' target='_blank' style='color: #a19eae; text-decoration: none;'><i class='fas fa-link'></i> Instagram Post</a>", unsafe_allow_html=True)

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)
                