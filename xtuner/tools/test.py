# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from types import FunctionType
import deepspeed
import h5py
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from sympy import im
from xtuner.model.titan.vision_tower.titan_vision_tower import TitanVisionTower
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import MAP_FUNC
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)
import torch
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from xtuner.utils import PROMPT_TEMPLATE
from PIL import Image
import pandas as pd
import numpy as np
from transformers import GenerationConfig, StoppingCriteriaList

import os

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument( '--vision_weight_path', default=None, help='path to  vision tower weight (.safetensors)')
    parser.add_argument('--llm_path',default=None,help='path to LLM (e.g., Qwen model directory)')
    parser.add_argument('--feature_dir', default=None, help='directory of test image features (.h5 files)')
    parser.add_argument('--test_slide_csv', default=None, help='test_slide_csv')
    parser.add_argument('--test_output_csv', default=None, help='test_output_csv')
    parser.add_argument(
    '--torch-dtype',
    default='bf16',
    choices=TORCH_DTYPE_MAP.keys(),
    help='Override the default `torch.dtype` and load the model under '
    'a specific `dtype`.')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

 
def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)

def main():
    torch.cuda.set_device(0)
    args = parse_args()

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register FunctionType object in cfg to `MAP_FUNC` Registry and
    # change these FunctionType object to str
    register_function(cfg._cfg_dict)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    
    model_kwargs = {
    'trust_remote_code': True,
    'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }

    state_dict = guess_load_checkpoint(args.checkpoint)

    runner.model.load_state_dict(state_dict, strict=False)
    runner.model.eval()
    runner.logger.info(f'Load checkpoint from {args.checkpoint}')


    if args.llm_path is None:
        raise ValueError("You must provide --llm_path")
    llm_name_or_path = args.llm_path
    
    tokenizer = AutoTokenizer.from_pretrained(
                llm_name_or_path,
                trust_remote_code=True,
                encode_special_tokens=True)

    llm = runner.model.llm
    llm.eval()

    projector = runner.model.projector.to(model_kwargs['torch_dtype'])
    projector.cuda()
    projector.eval()


    df_test_case = pd.read_csv(args.test_slide_csv)

    df_test_case['Output'] = df_test_case.apply(lambda x: '', axis=1)
    columns = ['metadata','question','image','T-answer','question_id','Output']
    df_test_output = pd.DataFrame(columns=columns)
    
    for i in range(df_test_case.shape[0]):
        
        print('*'*30)
        print('id: ', i)
        case_name = df_test_case.loc[i, 'image']
        if args.feature_dir is None:
            raise ValueError("You must provide --feature_dir")

        test_image_file = osp.join(args.feature_dir, str(case_name))
      


        if test_image_file.endswith('.h5'):
            with h5py.File(test_image_file, 'r') as h5_file:
                pixel_values = torch.from_numpy(h5_file['features'][:])
                coords = torch.from_numpy(h5_file['coords'][:])
                patch_size_lv0 = h5_file['coords'].attrs['patch_size_level0']
        if args.vision_weight_path is None:
            raise ValueError("You must provide --vision_weight_path")
        titan = TitanVisionTower(
            config_path="/home/liangyuci/SlideChat-titan/xtuner/model/titan/TITAN_local/config.json",
             weight_path=args.vision_weight_path
        )
        titan_output = titan(pixel_values, coords, patch_size_lv0)  
        titan_output = titan_output.unsqueeze(0)      
                      
                 
  
        image = titan_output.cuda()
        
        prompt_template = PROMPT_TEMPLATE.qwen_chat
        SYSTEM = ''
        question = df_test_case.loc[i, 'question']
        sample_input = f"{question}"
        print('Input: ', sample_input)

        instruction = prompt_template.get('INSTRUCTION', '{input}')
        sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
        inputs = (SYSTEM + instruction).format(input=sample_input, round=1, **runner.cfg)
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = tokenizer.encode(chunk)
            else:
                cur_encode = tokenizer.encode(
                    chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        input_ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            input_ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                input_ids.append(IMAGE_TOKEN_INDEX)
        input_ids = torch.tensor(input_ids).cuda()

        for name, param in runner.model.named_parameters():
            print(f"{name}: {param.dtype}")
            break  # 看第一个就够，先确认用的是什么精度
        for name, module in runner.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                print(f"{name} LayerNorm dtype: {module.weight.dtype}")


        print("当前模型 runner.model.llm 的数据类型是：", runner.model.llm.dtype)
        print(f"[调试] 原始 image.shape = {image.shape}, image.dtype = {image.dtype}")
        image = image.float()  # 防止 float64 → float16 报错
        image = image.to(torch.float16).cuda()
        print(f"[调试] 转换后 image.dtype = {image.dtype}")
        runner.model = runner.model.half()
     

        pixel_values = runner.model.projector(image)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=runner.model.llm,
            input_ids=input_ids.unsqueeze(0),
            pixel_values=pixel_values)

        max_new_tokens=500
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
        stop_words=[]
        stop_words += prompt_template.get('STOP_WORDS', [])
        stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            stop_criteria.append(
                StopWordStoppingCriteria(tokenizer, word))
        
        generate_output = llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            use_cache=False,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria)

        generation_output = tokenizer.decode(generate_output[0])
        if generation_output.endswith('<|im_end|>'):
            generation_output = generation_output[:-10]
            
        print('Output: ', generation_output)

        add_row = {
            'question_id': df_test_case.loc[i, 'question_id'],
            'image': df_test_case.loc[i, 'image'],
            'metadata': df_test_case.loc[i, 'metadata'],
            'question': question,
            'T-answer': df_test_case.loc[i, 'T-answer'],
            'Output': generation_output
        }
        df_test_output.loc[i] = add_row
        df_test_output.to_csv(args.test_output_csv)
        
    print('Test ok!')

if __name__ == '__main__':
    main()
