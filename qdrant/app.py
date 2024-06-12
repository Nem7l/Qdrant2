# Streamlit app
import base64
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient

collection_name = "instagram_posts"

if 'selected_record' not in st.session_state:
    st.session_state.selected_record = None

##############3

def select_record(new_record):
    st.session_state.selected_record = new_record

def get_init_records():
    client = get_client()
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=6,
    )
    return records

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
    return  QdrantClient(
        url="https://b088d693-20f0-4200-95c0-5ba21d0ace8c.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="uT8Jv8SlesDJLW0BEMS3t0JwZOjz_PthtzCVVaedMXLkKv2oI24MIQ",
    )


#### App
st.title("Qdrant Instagram Images")

records = get_init_records() if st.session_state.selected_record is None else get_similar_records()
if st.session_state.selected_record:
    image_bytes = get_bytes_from_base64(st.session_state.selected_record.payload["image"])
    st.image(image_bytes, use_column_width=True)
