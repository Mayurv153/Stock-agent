#!/bin/bash
# =============================================================
# Oracle Cloud VM — One-click deployment script
# Run this on your Oracle Cloud VM after SSH-ing in:
#   chmod +x deploy.sh && ./deploy.sh
# =============================================================

set -e

echo "=========================================="
echo "  AI Stock Agent — Oracle Cloud Setup"
echo "=========================================="

# 1. System updates
echo "[1/6] Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y

# 2. Install Python 3.12 + pip + git
echo "[2/6] Installing Python 3.12..."
sudo apt-get install -y python3.12 python3.12-venv python3-pip git

# 3. Clone repo
echo "[3/6] Cloning repository..."
cd /home/ubuntu
if [ -d "Stock-agent" ]; then
    cd Stock-agent && git pull
else
    git clone https://github.com/Mayurv153/Stock-agent.git
    cd Stock-agent
fi

# 4. Create virtual environment & install deps
echo "[4/6] Setting up Python environment..."
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create .env file (user must fill in values)
echo "[5/6] Setting up environment variables..."
if [ ! -f .env ]; then
    cat > .env << 'ENVEOF'
GROQ_API_KEY=your_groq_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password_here
EMAIL_RECEIVER=your_email@gmail.com
PORTFOLIO_CAPITAL=100000
ENVEOF
    echo ">>> IMPORTANT: Edit .env with your actual credentials!"
    echo ">>>   nano /home/ubuntu/Stock-agent/.env"
else
    echo ".env already exists, skipping..."
fi

# 6. Create required directories
echo "[6/6] Creating directories..."
mkdir -p reports logs templates

# 7. Install systemd service
echo "Installing systemd service..."
sudo cp stock-agent.service /etc/systemd/system/stock-agent.service
sudo systemctl daemon-reload
sudo systemctl enable stock-agent
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo "  1. Edit your credentials:"
echo "     nano /home/ubuntu/Stock-agent/.env"
echo ""
echo "  2. Start the agent:"
echo "     sudo systemctl start stock-agent"
echo ""
echo "  3. Check status:"
echo "     sudo systemctl status stock-agent"
echo ""
echo "  4. View logs:"
echo "     journalctl -u stock-agent -f"
echo "=========================================="
