TEST_PATH=~/datasets/seq34/transforms_one_test.json

python scripts/custom_evaluate.py \
    --mode nerf \
    --test_transforms $TEST_PATH