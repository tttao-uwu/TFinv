from pipelines.pipeline_inversion_sdturbo import StableDiffusionPipeline as StableDiffusionPipeline_update
from pipelines.sd_pipeline import StableDiffusionPipeline
from pipelines.scheduler_ddim import DDIMScheduler
from pipelines.scheduler_inv import DDIMInverseScheduler
from PIL import Image
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from diffusers.image_processor import VaeImageProcessor
import argparse
import pickle as pkl
import torch
import os



def get_latent_z_0(args, device=None, generator=None):
    
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32, variant="fp16")

    pipe = pipe.to("cuda")
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    input_image = Image.open(args.path_img_input).convert("RGB").resize((512, 512))
    
    # Encode with VAE
    with torch.no_grad():
        latent_z_0 = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
    z_0 = 0.18215 * latent_z_0.latent_dist.sample()
    
    
    if args.show_z_0:
        with torch.no_grad():
            image_rec = pipe.vae.decode(z_0.detach() / pipe.vae.config.scaling_factor, return_dict=False, )[0]
            image_rec = pipe.image_processor.postprocess(image_rec, output_type='pil', do_denormalize=[True])  
        
        path_image_rec = os.path.join(args.path_imgs_inversion, f"img_latent_z_0.jpg")
        (image_rec[0]).convert('RGB').save(path_image_rec)
        
    return z_0, pipe
       

@torch.no_grad()
def invert(
    pipe,
    start_latents,
    prompt,
    guidance_scale=0,
    num_inference_steps=10,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt=None,
    device=None):
    
    text_embeddings = pipe._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(0, num_inference_steps), total=num_inference_steps):
   
        # We'll skip the final iteration
        # if i >= num_inference_steps - 1:
        #     continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred
        # Store
        intermediate_latents.append(latents)
        
    return torch.cat(intermediate_latents)
    
 
    
def get_latent_z_T_ddim_inversion(args):
    
    model_id = args.model_id
    generator = torch.manual_seed(0)
    
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, variant="fp16")
    pipeline = pipeline.to("cuda")
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    image_processor=VaeImageProcessor()
    
    img_pth = args.path_img_input
    prompt = args.input_image_prompt
    
    _inv_raw_image_1 = Image.open(img_pth).convert("RGB").resize((512,512))

    inv_raw_image_1 = image_processor.preprocess(_inv_raw_image_1)
    
    
    latent = pipeline.prepare_image_latents(inv_raw_image_1.cuda(), 1, pipeline.vae.dtype, 'cuda',generator=generator)   
    # prompt_embeds, negative_embeds = pipeline.encode_prompt(prompt,device="cuda",num_images_per_prompt=1, do_classifier_free_guidance=False)
    
    # ###inversion
    
    pipeline.scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.safety_checker=None


    latent_z_T_ddim_inversion = pipeline(
                                        # prompt_embeds=prompt_embeds, 
                                        prompt=prompt, 
                                        # negative_prompt_embeds=negative_embeds, 
                                        generator=generator, 
                                        num_inference_steps=args.inference_steps_DDIM_inversion,
                                        # output_type="pt", ### NOTE: can be pil  latent
                                        output_type="latent", ### NOTE: can be pil  latent
                                        latents=latent,
                                        guidance_scale=0.0,
                                        )
    output_latent = latent_z_T_ddim_inversion.images
    
    with torch.no_grad():
        image_rec = pipeline.vae.decode(output_latent.detach() / pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image_rec = pipeline.image_processor.postprocess(image_rec, output_type='pil', do_denormalize=[True])

    ### NOTE: save the recon images
    # caption = args.concept_learning_prompt
    path_image_rec = os.path.join(args.path_imgs_inversion, f"img_latent_z_T_DDIM_inversion.jpg")
    (image_rec[0]).convert('RGB').save(path_image_rec)
        
    
    return latent_z_T_ddim_inversion.images

    
    
def verify_latent_z_0_sample(args, latent):
    
    init_latent = latent.detach().clone()
    init_latent = torch.nn.Parameter(init_latent,)
    generator = torch.manual_seed(0)
    prompt_ = args.concept_learning_prompt
    model_id = args.model_id
    # pipeline
    pipeline = StableDiffusionPipeline_update.from_pretrained(model_id, torch_dtype=torch.float32, variant="fp16")
    pipeline = pipeline.to("cuda")

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    ### latent is the target
    ### NOTE: note that I only tested the num_inference_steps as 1. You should accumulate the grad for this as 2 to 4
    ### NOTE: Reconstruction

    with torch.no_grad():
        output_rec_1 = pipeline.rec_images(
                            prompt=prompt_, 
                            # negative_prompt_embeds=negative_embeds, 
                            generator=generator, 
                            num_inference_steps=args.inference_steps_samlpe_DDIM_inversion,
                            # output_type="pt", ### NOTE: can be pil  latent
                            output_type="latent", ### NOTE: can be pil  latent
                            latents=init_latent,
                            guidance_scale=0.0,
                            )
        

        output_latent = output_rec_1.images

    ### NOTE: updated init_latent as the initial noise

    generator = torch.manual_seed(0)
    with torch.no_grad():
        image_rec = pipeline.vae.decode(output_latent.detach() / pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image_rec = pipeline.image_processor.postprocess(image_rec, output_type='pil', do_denormalize=[True])

    ### NOTE: save the recon images
    # caption = args.concept_learning_prompt
    path_image_rec = os.path.join(args.path_imgs_inversion, f"img_latent_z_T_rec_direct_sample.jpg")
    (image_rec[0]).convert('RGB').save(path_image_rec)
        


    
    
def get_latent_z_T_init(args, pipeline, z_0, device, generator):
    
    inverted_latents = invert(pipe=pipeline, 
                              start_latents=z_0, 
                              prompt=args.input_image_prompt, 
                              num_inference_steps=args.inference_steps_get_init_latents, 
                              device=device)
    init_latent = inverted_latents[-1][None]
    
    if args.show_z_T_init:
        
        with torch.no_grad():
            image_rec = pipeline.vae.decode(init_latent.detach() / pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image_rec = pipeline.image_processor.postprocess(image_rec, output_type='pil', do_denormalize=[True])  
    
        caption = args.concept_learning_prompt
        path_image_rec = os.path.join(args.path_imgs_z_T_init, f"img_latent_z_T_init_{caption}.jpg")
        (image_rec[0]).convert('RGB').save(path_image_rec)

    
    return init_latent
    


def get_pipeline_update_inversion(args):

    pipeline_text2image_update = StableDiffusionPipeline_update.from_pretrained(args.model_id,
                                                                            torch_dtype=torch.float32,
                                                                            variant="fp16")
    pipeline_text2image_update = pipeline_text2image_update.to("cuda")

    pipeline_text2image_update.scheduler = DDIMScheduler.from_config(pipeline_text2image_update.scheduler.config)

    return pipeline_text2image_update



def get_latents_update(args, pipeline_update, latents_target, init_latents, generator):
    
    tokenizer = pipeline_update.tokenizer   # ### len(tokenizer) = 49408
    
    # ### NOTE: len(tokenizer) = 49408+6
    num_added_tokens = tokenizer.add_tokens(args.placeholder_tokens)
    
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(args.placeholder_tokens)

    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)

    initializer_token_id = token_ids

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder = pipeline_update.text_encoder
    text_encoder.resize_token_embeddings(len(tokenizer))    # ### len(tokenizer) = 49408 + num_added_tokens
    
    # Initialise the newly added placeholder token with the embeddings of the initializer token
    # ### token_embeds shape = [49408+num_added_tokens,1024]
    token_embeds = text_encoder.get_input_embeddings().weight.data  
    tmp_count=0
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id[tmp_count]].clone()
            tmp_count = tmp_count + 1
    
    
    index_no_updates = torch.ones(len(tokenizer), dtype=bool)  # len(index_no_updates) = 49408+num_added_tokens
    
    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
    
    for ind in range(len(placeholder_token_ids)):

        index_no_updates[placeholder_token_ids[ind]]=False
    print(text_encoder)
    # Freeze all parameters except for the token embeddings in text encoder 
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    # Freeze vae and unet
    pipeline_update.vae.requires_grad_(False)
    pipeline_update.unet.requires_grad_(False)
    
    
    with torch.enable_grad():

        img_update, latents_update, img_latent_z_T, tokens_concept = pipeline_update(
                                    prompt=args.concept_learning_prompt,
                                    generator=generator,
                                    placeholder_token_id=placeholder_token_ids, 
                                    placeholder_tokens = args.placeholder_tokens,
                                    num_inference_steps=args.inference_steps_get_update_inversion,
                                    output_type="latent",
                                    latents_target=latents_target.cuda(),
                                    latents=init_latents,
                                    guidance_scale=0.0,
                                    update_step_num=args.inversion_update_steps,
                                    index_no_updates=index_no_updates,
                                    path_image_cross_attn_inversion = args.path_imgs_cross_attn_inversion,
                                    path_image_self_attn_inversion = args.path_imgs_self_attn_inversion,
                                    path_image_distribution = args.path_imgs_distribution,
                                    path_image_input=args.path_img_input,
        )

    path_save_latents = os.path.join(args.path_latents_update, f"latents.pkl")
    path_save_tokens = os.path.join(args.path_tokens_update, f"token.pkl")
    
    if args.show_z_T_update:
        
        caption = args.concept_learning_prompt
        path_latents_update_image = os.path.join(args.path_imgs_z_T_update, f"img_latent_z_T_update_{caption}.jpg")
        img_latent_z_T[0].save(path_latents_update_image)
    
    
    with open(path_save_latents, 'wb') as f:
        pkl.dump(latents_update, f)
    
    with open(path_save_tokens, 'wb') as f:
        pkl.dump(tokens_concept, f)


    pipeline_update.scheduler = DDIMScheduler.from_config(pipeline_update.scheduler.config)
    
    if args.show_image_update:
        
        with torch.no_grad():
            image_update = pipeline_update.vae.decode(
                img_update.images.detach() / pipeline_update.vae.config.scaling_factor,
                return_dict=False, generator=generator)[0]

            image_update = pipeline_update.image_processor.postprocess(image_update, output_type='pil', do_denormalize=[True])
            
        
        caption = args.concept_learning_prompt
        path_image_update = os.path.join(args.path_imgs_inversion,f"img_update_{caption}.jpg")
        
        (image_update[0]).convert('RGB').save(path_image_update)
    
    
    return pipeline_update



def arguments():
    parser = argparse.ArgumentParser()
    
    
    
    parser.add_argument('--model_id', type=str, default='stabilityai/sd-turbo')
    parser.add_argument('--path_img_input', type=str, default='')

    
    parser.add_argument('--path_imgs_inversion', type=str, default='')
    parser.add_argument('--path_imgs_z_T_init', type=str, default='')
    parser.add_argument('--path_imgs_z_T_update', type=str, default='')
    
    parser.add_argument('--path_latents_update', type=str, default='')
    parser.add_argument('--path_tokens_update', type=str, default='')
        
    parser.add_argument('--path_imgs_cross_attn_inversion', type=str, default='')
    parser.add_argument('--path_imgs_self_attn_inversion', type=str, default='')
    parser.add_argument('--path_imgs_distribution', type=str, default='')    
    
    parser.add_argument('--input_image_prompt', type=str, default="")
    # parser.add_argument('--negative_prompt', type=str, default=None)


    parser.add_argument('--placeholder_tokens', nargs='+', type=str, default=[])
    parser.add_argument('--initializer_token', nargs='+', type=str, default=[])
    parser.add_argument('--concept_learning_prompt', nargs='+', type=str, default=[""])
    
    parser.add_argument('--inversion_update_steps', type=int, default='')
    parser.add_argument('--inference_steps_get_init_latents', nargs='+', type=int, default='')
    parser.add_argument('--inference_steps_get_update_inversion', nargs='+', type=int, default='')
    # ###NOTE: DDIM 
    # parser.add_argument('--inference_steps_DDIM_inversion', nargs='+', type=int, default='30')
    # parser.add_argument('--inference_steps_samlpe_DDIM_inversion', nargs='+', type=int, default='1')
    
    
    # if show image
    parser.add_argument('--show_z_0', type=bool, default=False)
    parser.add_argument('--show_z_T_init', type=bool, default=False)
    parser.add_argument('--show_image_update', type=bool, default=False)
    parser.add_argument('--show_z_T_update', type=bool, default=False)
    
    
    
    args = parser.parse_args()
    return args


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = torch.manual_seed(0)
    
    args = arguments()
    
    latent_z_0, pipeline = get_latent_z_0(args=args, device=device, generator=generator)
    
    latent_z_T_init = get_latent_z_T_init(args=args, pipeline=pipeline, z_0=latent_z_0,  device=device, generator=generator)

    # # NOTE: DDIM 
    # latent_z_T_init_ddim_sample = get_latent_z_T_ddim_inversion(args)
    
    # verify_latent_z_0_sample(args, latent_z_T_init_ddim_sample)
    
    
    pipe_update_inversion = get_pipeline_update_inversion(args)
    
    # latent_z_0_init = latent_z_0.clone()
    
    pipeline_update = get_latents_update(args=args,
                                pipeline_update=pipe_update_inversion,  
                                latents_target=latent_z_0, 
                                init_latents=latent_z_T_init,
                                generator=generator)
