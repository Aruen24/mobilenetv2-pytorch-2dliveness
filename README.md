python forward.py --batch_size=1 --use_half=1 --quantized_mode='int8' --core_num 1
python forward.py --batch_size=1 --use_half=1 --quantized_mode='int16' --core_num 1
# cnrtexec ghostnet_int8_b4_c4.cambricon 0 5000 4 1
