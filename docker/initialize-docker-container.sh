#!/bin/bash
source ~/.bashrc
#install pi0
cd ~/lerobot && pip install -e ".[pi0]"
pip install pytest

# git safe directory
cd ~/lerobot && git config --global --add safe.directory /root/lerobot

cd ~/lerobot

echo "Finished setting up container"

# https://stackoverflow.com/questions/30209776/docker-container-will-automatically-stop-after-docker-run-d
tail -f /dev/null

