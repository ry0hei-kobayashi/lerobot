singularity run --nvccli \
    --bind /tmp/.X11-unix:/tmp/.X11-unix \
    pi0.sif
    #--bind ../pg-blip:/pg-blip \
    #--bind ../models/pgvlm_weights.bin:/models/pgvlm_weights.bin \
    #--bind ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \

#--nv --no-home \
