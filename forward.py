import torch
import sys
import cv2
import os
import torch_mlu
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct
import numpy as np
from model import mobileNetv2
import time
import argparse
parser = argparse.ArgumentParser(description='')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
      
parser.add_argument('--core_num', default=1, type=int, help='core num')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--use_half', default=False, type=str2bool, help='use_half')
parser.add_argument('--quantized_mode', default='int8', type=str, help='(int8,int16)')


args = parser.parse_args()



def quantification(quanti_type):
    print("======start quantification======")
    imgfile = './data/mask/mask.jpg'
    #imgfile = './test/fake_1872.png'
    batch_size = 1
    times = 1

    with torch.no_grad():
        # mean = [0.5, 0.5, 0.5]
        # std = [0.5, 0.5, 0.5]
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        model = mobileNetv2()
        print("======load model sucess======")
        model.eval().float()
        model.load_state_dict(torch.load('./models/model_new.pth', map_location='cpu'), strict=False)
        print("======load pth sucess======")
        model = model.to(torch.device('cpu'))
        model = mlu_quantize.quantize_dynamic_mlu(model, {'iteration':times, 'mean': mean, 'std': std, 'data_scale':1.0, 'perchannel':False, 'use_avg':False,'firstconv':True }, dtype=quanti_type, gen_quant=True )
        for i in range(times):
            #imgfile = image_path.format(100 + i)
            
            img = cv2.imread(imgfile)
            sized = cv2.resize(img, (112, 112)).T
            #sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            sized = sized / 255    #cpu上执行推理
            
            #sized = np.transpose(sized, (2, 0, 1))
            sized = sized.astype(np.float32)
            input_img = torch.from_numpy(np.stack([sized]*batch_size))
            print("quantification###############")
            print(input_img.shape)  # 1 3 112 112
            print(input_img)
            outs = model(input_img)
            print(outs.cpu().float())
        save_path = './models/mobilev2_{}.pth'.format(quanti_type)
        torch.save(model.state_dict(), save_path)
        print("======end quantification======")

def do_detect(model, img,  use_cuda=0, use_mlu=1, half_input=0):
    model.eval()
    t0 = time.time()
    
    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        if use_mlu:
            print('using mlu**********')
            #input_data = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            input_data = torch.from_numpy(img).float().unsqueeze(0)
            print(input_data.shape)
            print(input_data)
        else:
            input_data = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        input_data = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        input_data = input_data.cuda()

    #input_data = torch.autograd.Variable(input_data)

    if use_mlu:
        if half_input:
            input_data = input_data.type(torch.HalfTensor)
        input_data = input_data.to(ct.mlu_device())

    t1 = time.time()

    outputs = model(input_data)
    # print('outputs size:{}'.format(outputs.shape))
    t2 = time.time()
    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')

    if use_mlu:
        print(outputs.shape)
        outputs = outputs.cpu().float()
        return outputs
    else:
        return outputs#post_processing(input_data, conf_thresh, nms_thresh, outputs)

def forward(imgfile, use_mlu = 1, fusion = 0, half_input=0, dtype='int8',batch_size=1):
    #ct.set_core_version("MLU270")
    ct.set_core_version("MLU270")
    ct.set_core_number(args.core_num)
    if use_mlu:
        device = ct.mlu_device()
    else :
        device = torch.device('cpu')

    intx_pth_path = './models/mobilev2_{}.pth'.format(dtype)
    if not os.path.exists(intx_pth_path):
        quantification(dtype)
       
    img = cv2.imread(imgfile)
    model_name = './models/mobilev2_{}.pth'.format(dtype)
    with torch.no_grad():
        model = mobileNetv2()
        model.eval().float()
        model = mlu_quantize.quantize_dynamic_mlu(model)
        model.load_state_dict(torch.load(model_name), strict=False)
        model.to(device)

        # preprocess
        sized = cv2.resize(img, (112, 112)).T
        #sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        sized = sized   #调用量化后模型
        input_h = sized.shape[2]
        input_w = sized.shape[1]
        print('H:{}, W:{}'.format(input_h, input_w))
        trace_input = torch.randn((batch_size, 3, input_w, input_h), dtype=torch.float)
        # fusion mode
        if fusion:
            offline_name = './result/mobilev2_{}_b{}_c{}'.format(dtype,batch_size,args.core_num)
            if half_input:
                print('half input')
                trace_input = trace_input.type(torch.HalfTensor)
                print(trace_input.shape)
                print(trace_input)
            ct.save_as_cambricon(offline_name)
            model = torch.jit.trace(model, trace_input.to(device), check_trace=False)
            print('fusion success')
            out = model(trace_input.to(device))
            ct.save_as_cambricon('')
        boxes = do_detect(model, sized,  0, use_mlu, half_input)
        print(boxes)


def main():
    # quantification('int8')
    imgfile = './data/mask/mask.jpg'
    #imgfile = './test/test2.png'
    use_mlu = 1
    batch_size=args.batch_size
    fusion=1
    half_input=args.use_half
    dtype=args.quantized_mode
    forward(imgfile, use_mlu, fusion, half_input,dtype,batch_size)

if __name__ == '__main__':
    main()

