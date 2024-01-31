## Test with Nguyens weights

# For preparing the model with the trained weights
cd <PROJECT_ROOT>
mkdir -R ./out/runs/cifar10-cosine-s0-e1-tau1 && cd "$_"
echo '{"time_steps": 1000, "schedule": { "type": "cosine", "start": 0.0, "end": 1.0, "tau": 1.0 } }' > diffuser.config.json
echo '{"image_size": 32, "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"], "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"], "layers_per_block": 2, "time_embedding_type": "positional", "dropout_rate": 0.1}' > model.config.json
cp <PATH/TO/WEIGHTS.pt> ./weights.pt
# For generating images
cd <PROJECT_ROOT>
python3 generate_images.py ./out/runs/cifar10-cosine-s0-e1-tau1 100 --batch-size 10

## Model selection

# General parameters
# T = 1000
# N = 6.250
# train_split = 0.8 => N_train = 5.000
# test_split = 0.2  => N_test  = 1.250
# epochs = 100 => total iterations: 500k (train) + 125k (test) = 625k
# batch_size = 32
# learning_rate = 0.0002 (2 * 1e-4)
# dropout_rate = 0.1

cd <PROJECT_ROOT>

# Model 1: Linear scheduler with the same parameters as Ho et. al (https://arxiv.org/pdf/2301.10972.pdf)
# linear, s=0.0001, e=0.02
python3 train_denoiser.py cifar10 \
    --image-size 32 \
    --time-steps 1000 \
    --epochs 100 \
    --subset-size 6250 \
    --train-split 0.8 \
    --batch-size 32 \
    --learning-rate 0.0002 \
    --dropout-rate 0.1 \
    --schedule linear \
    --schedule-start 0.0001 \
    --schedule-end 0.02

# DONE ALREADY: Model 2: Cosine scheduler with standard parameters
# cosine, s=0.0, e=1.0, tau=1.0
# -

# Model 3: Cosine scheduler with best parameters for 64x64 according to Ting Chen (https://arxiv.org/pdf/2301.10972.pdf)
# cosine, s=0.2, e=1.0, tau=1.0
python3 train_denoiser.py cifar10 \
    --image-size 32 \
    --time-steps 1000 \
    --epochs 100 \
    --subset-size 6250 \
    --train-split 0.8 \
    --batch-size 32 \
    --learning-rate 0.0002 \
    --dropout-rate 0.1 \
    --schedule cosine \
    --schedule-start 0.2 \
    --schedule-end 1.0 \
    --schedule-tau 1.0

# Model 4: Sigmoid scheduler with best parameters for 64x64 according to Ting Chen (https://arxiv.org/pdf/2301.10972.pdf)
# sigmoid, s=-3.0, e=3.0, tau=1.1
python3 train_denoiser.py cifar10 \
    --image-size 32 \
    --time-steps 1000 \
    --epochs 100 \
    --subset-size 6250 \
    --train-split 0.8 \
    --batch-size 32 \
    --learning-rate 0.0002 \
    --dropout-rate 0.1 \
    --schedule sigmoid \
    --schedule-start -3.0 \
    --schedule-end 3.0 \
    --schedule-tau 1.1

# OPTIONAL: Model 5: Sigmoid scheduler with best parameters for 128x128 according to Ting Chen (https://arxiv.org/pdf/2301.10972.pdf)
# sigmoid, s=0.0, e=3.0, tau=0.9
python3 train_denoiser.py cifar10 \
    --image-size 32 \
    --time-steps 1000 \
    --epochs 100 \
    --subset-size 6250 \
    --train-split 0.8 \
    --batch-size 32 \
    --learning-rate 0.0002 \
    --dropout-rate 0.1 \
    --schedule sigmoid \
    --schedule-start 0.0 \
    --schedule-end 3.0 \
    --schedule-tau 0.9
