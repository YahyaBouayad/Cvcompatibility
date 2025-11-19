# pages/01_📋_Jobs_et_Candidats.py
from __future__ import annotations

import json
import os
from datetime import datetime, date
from typing import Optional, Any, Iterable, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import requests

# Azure Blob
try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except Exception:
    BlobServiceClient = None

st.set_page_config(page_title="Jobs publiés & candidats — vue cartes", page_icon="📋", layout="wide")
st.title("📋 Jobs publiés & candidats — vue cartes")