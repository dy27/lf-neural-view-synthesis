TRAIN_PATH=~/datasets/seq34/transforms_train_dense.json
TEST_PATH=~/datasets/seq34/transforms_single_test.json
N_STEPS=2000

python scripts/custom_run.py \
    --mode nerf \
    --scene $TRAIN_PATH \
    --test_transforms $TEST_PATH \
    --n_steps $N_STEPS \