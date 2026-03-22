from pipelines.pipeline_edit_sdturbo import StableDiffusion_EditPipeline
from pipelines.scheduler_ddim import DDIMScheduler
from pipelines.scheduler_inv import DDIMInverseScheduler
from IPython.display import display
from PIL import Image
from typing import List, Tuple, Union
from torchvision import transforms as tfms
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from diffusers.utils import pt_to_pil
from diffusers.image_processor import VaeImageProcessor
import argparse
import numpy as np
import pickle as pkl
import torch
import datetime
import os
import cv2
from torchvision import utils as vutils
from torchvision.transforms import ToPILImage

from utils import (save_cross_attn, 
                   show_cross_attention_plus_orig_img, 
                   show_cross_attention, 
                   show_cross_attention_blackwhite, 
                   show_image_relevance, 
                   view_images, 
                   text_under_image,
                   show_image_distribution,
                   draw_pca,
                   )



def get_image_edit(pipeline, args, generator, latents_update, tokens_embds_learn):
    
    tokenizer = pipeline.tokenizer   # ### len(tokenizer) = 49408
    
    # ### NOTE: len(tokenizer) = 49408+num_added_tokens
    num_added_tokens = tokenizer.add_tokens(args.placeholder_tokens)
    
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(args.placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder = pipeline.text_encoder
    text_encoder.resize_token_embeddings(len(tokenizer))    # ### len(tokenizer) = 49408 + num_added_tokens
    
    # Initialise the newly added placeholder token with the embeddings of the initializer token
    # ### token_embeds shape = [49408+num_added_tokens,1024]
    token_embeds = text_encoder.get_input_embeddings().weight.data  
    tmp_count=0
    with torch.no_grad():
        for token_id in placeholder_token_ids:
       
            token_embeds[token_id] = tokens_embds_learn[tmp_count]
            tmp_count = tmp_count + 1
    
    # Freeze all parameters except for the token embeddings in text encoder 
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # Freeze vae and unet
    pipeline.vae.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    with torch.no_grad():
        img_new_prompt = pipeline(prompt=args.caption, 
                                edit_prompt=args.edit_caption,                 
                                latents=latents_update, 
                                token_indices=args.indices_to_alter,
                                output_type="latent",
                                num_inference_steps=args.inference_steps_get_edit_image, 
                                guidance_scale=0, 
                                generator=generator,
                                refine=args.refine,
                                replace=args.replace,
                                local=args.local,
                                mapper = args.mapper, 
                                alphas = args.alphas,
                                cross_replace_steps=args.cross_replace_steps,
                                self_replace_steps=args.self_replace_steps,
                                indices_to_amplify=args.indices_to_amplify,
                                amplify_scale=args.amplify_scale)
        
        image_new_prompt = pipeline.vae.decode(
            img_new_prompt.images.detach() / pipeline.vae.config.scaling_factor,
            return_dict=False, generator=generator)[0]
        
        # image_rec = torch.unsqueeze(image_new_prompt[0],dim=0)
        # img_rec = pipeline.image_processor.postprocess(image_rec, output_type='pil', do_denormalize=[True])
        # prompt = args.caption
        # path_p2p_rec = os.path.join(args.path_imgs_p2p, f"img_rec_{prompt}.jpg")
        # (img_rec[0]).convert('RGB').save(path_p2p_rec)
        
        if args.show_edit_image:
            image_edit = torch.unsqueeze(image_new_prompt[1],dim=0)
            img_edit = pipeline.image_processor.postprocess(image_edit, output_type='pil', do_denormalize=[True])
            edit_prompt = args.edit_caption
            path_p2p_edit = os.path.join(args.path_imgs_p2p, f"{edit_prompt}.jpg")
            (img_edit[0]).convert('RGB').save(path_p2p_edit)
        
        return pipeline
        # return pipeline, self_atten_raw, cross_atten_raw, self_atten_edit, cross_atten_edit
        
        
    
def arguments():
    parser = argparse.ArgumentParser()
    
    # a cup of coffee with drawing of lion putted on the wooden table
    # a dog sitting on a wooden chair
    
    parser.add_argument('--model_id', type=str, default='stabilityai/sd-turbo')
    parser.add_argument('--path_img_input', type=str, default='')
    
    parser.add_argument('--path_latents_update', type=str, default='')
    parser.add_argument('--path_tokens_update', type=str, default='')
    
    parser.add_argument('--path_attention_imgs_p2p', type=str, default='p')
    parser.add_argument('--path_imgs_p2p', type=str, default='')
    
    parser.add_argument('--negative_prompt', type=str, default=None)
    parser.add_argument('--placeholder_tokens', nargs='+', type=str, default=[])
    parser.add_argument('--caption', nargs='+', type=str, default='')
    parser.add_argument('--edit_caption', nargs='+', type=str, default='')
    parser.add_argument('--inference_steps_get_edit_image', nargs='+', type=int, default='1')
    
    parser.add_argument('--indices_to_alter', nargs='+', type=int, default=None)
    
    parser.add_argument('--refine', action='store_true', default=False)
    parser.add_argument('--replace', action='store_true', default=False)
    parser.add_argument('--cross_replace_steps', nargs='+', type=int, default=1)
    parser.add_argument('--self_replace_steps', nargs='+', type=int, default=1)
    parser.add_argument('--indices_to_amplify', nargs='+', type=int, default=[1, 2])
    parser.add_argument('--amplify_scale', nargs='+', type=float, default=[1.0, 1.0])
    
    parser.add_argument('--local', default=False)
    parser.add_argument('--mapper', nargs='+', type=float, default=None)
    parser.add_argument('--alphas', nargs='+', type=float, default=None)
    
    parser.add_argument('--show_edit_image', type=bool, default=True)
    
    args = parser.parse_args()
    return args





if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"        
    args = arguments()
    generator_a = torch.manual_seed(0)
    print(args)
    
    with open(os.path.join(args.path_latents_update, f"latents.pkl"), 'rb') as f:
        
        inv_latents= pkl.load(f)
    
    with open(os.path.join(args.path_tokens_update, f"token.pkl"), 'rb') as ff:
        
        token_embeds_learn= pkl.load(ff)
        
            
    pipeline_p2p = StableDiffusion_EditPipeline.from_pretrained(args.model_id,
                                                                torch_dtype=torch.float32,
                                                                variant="fp16")
    pipeline_p2p = pipeline_p2p.to("cuda")


    pipeline  = get_image_edit(pipeline=pipeline_p2p, 
                                args=args, 
                                generator=generator_a, 
                                latents_update=inv_latents, 
                                tokens_embds_learn=token_embeds_learn)

    # get_raw_cross_attn_maps(args=args, cross_attn_maps =cross_attns_raw, spec='raw')
    
    # get_edit_cross_attn_maps(args=args, cross_attn_maps =cross_attns_edit, spec='edit')
    
    # get_raw_self_attn_maps(args=args, self_attn_maps =self_attns_raw, spec='raw')
    
    # get_edit_self_attn_maps(args=args, self_attn_maps =self_attns_edit, spec='edit')
    
    