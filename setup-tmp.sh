echo "Preparing environment for agentgym-rl..."
# conda create -n agentgym-rl python==3.10 -y
# conda activate agentgym-rl
# pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# install flash-atten
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_URL2="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME2="flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip3 install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME
# for RL
cd AgentGym-RL
pip3 install -e .
# for agentgym
echo "Preparing environment for agentenv..."
cd AgentGym/agentenv
pip3 install -e .
pip3 install transformers==4.51.3