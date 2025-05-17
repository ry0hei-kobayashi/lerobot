
if [ -n "${HF_TOKEN:-}" ]; then
  echo 'HF_TOKEN has already set'
else
  echo 'ERROR: Please set HF_TOKEN at first'
  exit 1
fi

singularity run --nv \
    --bind /tmp/.X11-unix:/tmp/.X11-unix \
    --bind ~/vla/lerobot:/root/lerobot \
    pi0.sif
