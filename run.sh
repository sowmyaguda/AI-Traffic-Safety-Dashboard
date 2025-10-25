#!/bin/bash

# Interactive CV Dashboard Launcher
# Uses RAPIDS environment with GPU acceleration

echo "Starting Interactive CV Analytics Dashboard"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Activate RAPIDS environment
echo "Activating RAPIDS environment..."
source ~/miniconda3/bin/activate rapids-25.10

if [ $? -ne 0 ]; then
    echo "Error: Could not activate rapids-25.10 environment"
    echo "   Run: conda env list"
    exit 1
fi

echo "RAPIDS environment activated"
echo ""

# Install requirements
echo "Installing/updating requirements..."
pip install -q -r requirements.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup complete!"
echo ""
echo "Before using:"
echo "   1. Get your FREE Groq API key from:"
echo "      https://console.groq.com/"
echo ""
echo "   2. Enter it in the sidebar when the app opens"
echo ""
echo "   3. Upload your CSV files (vehicle and crash data)"
echo ""
echo "   4. Start chatting! Examples:"
echo "      - 'Show speeding on I-70'"
echo "      - 'Filter by speed over 80'"
echo "      - 'Show crash hotspots'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Launching dashboard on port 8501..."
echo "   Access at: http://localhost:8501"
echo ""

# Run Streamlit
streamlit run app.py --server.port 8501 --server.address localhost --server.baseUrlPath /cv-dashboard
