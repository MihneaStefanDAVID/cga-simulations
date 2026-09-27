#!/bin/bash
# Double-click (macOS) to open the cGA simulator UI in the browser.
cd "$(dirname "$0")"
python3 -m streamlit run app.py
