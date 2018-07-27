sudo pip install numpy
python train.py --training-path "s3://testbucket/jarvis-data/zhuxunjie01_sx/train/" \
--batch-size 128 \
--model-path "s3://testbucket/jarvis/zhuxunjie01_sx/" \
--run-valid-every 10000 --save-model-every 200000 \
--num-epoches 1000