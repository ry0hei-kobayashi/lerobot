
if [ -n "${HF_TOKEN:-}" ]; then
  echo 'HF_TOKEN has already set'
else
  echo 'ERROR: Please set HF_TOKEN at first'
  exit 1
fi

#singularity run --nvccli \
singularity run --nv \
    --bind /tmp/.X11-unix:/tmp/.X11-unix \
    --bind ~/vla/lerobot:/root/lerobot \
    pi0.sif \
    python /root/lerobot/lerobot/scripts/eval.py --pretrained_policy.path=/root/lerobot/lerobot/apptainer/outputs/train/2025-05-18/02-33-56_pi0/checkpoints/last/pretrained_model

#echo "train starting"
#python /root/lerobot/lerobot/scripts/train.py \
#    --policy.path=lerobot/pi0 \
#    --dataset.repo_id=danaaubakirova/koch_test
#echo "train finished"

