import os
import torch
import gc
from ..utils import log, print_memory, fourier_filter
import math
from tqdm import tqdm

from ..wanvideo.modules.model import rope_params
from ..wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from ..wanvideo.utils.scheduling_flow_match_lcm import FlowMatchLCMScheduler
from ..wanvideo.utils.basic_flowmatch import FlowMatchScheduler
from ..nodes import optimized_scale
from einops import rearrange

from ..enhance_a_video.globals import disable_enhance

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
from comfy.clip_vision import clip_preprocess, ClipVisionModel
from comfy.cli_args import args, LatentPreviewMethod

script_directory = os.path.dirname(os.path.abspath(__file__))


#region Sampler
class WanVideoCausVidSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS", ),
                "image_embeds": ("WANVIDIMAGE_EMBEDS", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "shift": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": ([
                    "flowmatch_causvid", "flowmatch_causvid_14b", "flowmatch_causvid_self_forcing",
                    #"unipc", "unipc/beta", "euler", "euler/beta", "lcm", "lcm/beta"
                    ],
                    {
                        "default": 'flowmatch_causvid'
                    }),
                "kv_cache_device": (["main_device", "offload_device"], {"default": "offload_device", "tooltip": "Device to cache to"}),

            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "prefix_samples": ("LATENT", {"tooltip": "prefix latents"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rope_function": (["default", "comfy"], {"default": "default", "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile"}),
                "experimental_args": ("EXPERIMENTALARGS", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def _initialize_kv_cache(self, batch_size, dtype, device, num_blocks=30, num_heads=12):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(num_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.cache_window_size, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.cache_window_size, num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device, num_blocks=30, num_heads=12):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(num_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_heads, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
    
    def _shift_kv_cache(self):
        """
        Shift the KV cache left by shift_blocks * num_frame_per_block * frame_seq_length.
        This is called when kv_start exceeds window_size.
        The first block is preserved, and shifting starts from the second block.
        """
        shift_length = self.shift_blocks * self.num_frame_per_block * self.frame_seq_length
        
        for block in self.kv_cache1:            
            block["k"] = torch.roll(block["k"], shifts=-shift_length, dims=1)
            block["v"] = torch.roll(block["v"], shifts=-shift_length, dims=1)
            
            # Clear the shifted-out part (except the first block)
            block["k"][:, -shift_length:] = 0
            block["v"][:, -shift_length:] = 0
        
        # Update kv_start
        self.kv_start -= shift_length
        return shift_length
    

    def process(self, model, text_embeds, image_embeds, shift, steps, seed, scheduler, kv_cache_device,
        force_offload=True, samples=None, prefix_samples=None, denoise_strength=1.0, rope_function="default", 
        experimental_args=None):
        #assert not (context_options and teacache_args), "Context options cannot currently be used together with teacache."
        patcher = model
        model = model.model
        transformer = model.diffusion_model
        dtype = model["dtype"]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if kv_cache_device == "main_device":
            cache_device = mm.get_torch_device()
        else:
            cache_device = mm.unet_offload_device()
        
        steps = int(steps/denoise_strength)

        timesteps = None
        if 'unipc' in scheduler:
            sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        elif 'euler' in scheduler:
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
            sample_scheduler.set_timesteps(steps, device=device)
        elif 'lcm' in scheduler:
            sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
            sample_scheduler.set_timesteps(steps, device=device)
        elif 'flowmatch_causvid' in scheduler:
            sample_scheduler = FlowMatchScheduler(
                shift=shift, sigma_min=0.0, extra_one_step=True
            )
            sample_scheduler.set_timesteps(1000, training=True)
            denoising_step_list = torch.tensor([1000, 757, 522], dtype=torch.long)
          
            if "warp" in scheduler or "self_forcing" in scheduler:
                denoising_step_list = torch.tensor([1000, 750, 500, 250] , dtype=torch.long)
                timesteps = torch.cat((sample_scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                denoising_step_list = timesteps[1000 - denoising_step_list]
            elif "14b" in scheduler:
                denoising_step_list = torch.tensor([1000, 934, 862, 756, 603, 410, 250, 140, 74], dtype=torch.long)
            # sample_scheduler = FlowMatchScheduler(num_inference_steps=steps, shift=shift, sigma_min=0, extra_one_step=True)
            # sample_scheduler.timesteps = torch.tensor(denoising_step_list).to(device)
            # sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
            #print(sample_scheduler.sigmas)
            
        
        timesteps = denoising_step_list
        #timesteps = torch.tensor(denoising_list).to(device)
        print("timesteps", timesteps)
        
        if denoise_strength < 1.0:
            steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1):] 
        
        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)
       
        clip_fea, clip_fea_neg = None, None
        vace_data, vace_context, vace_scale = None, None, None

        image_cond = image_embeds.get("image_embeds", None)

        target_shape = image_embeds.get("target_shape", None)
        if target_shape is None:
            raise ValueError("Empty image embeds must be provided for T2V (Text to Video")
        
        has_ref = image_embeds.get("has_ref", False)
        vace_context = image_embeds.get("vace_context", None)
        vace_scale = image_embeds.get("vace_scale", None)
        vace_start_percent = image_embeds.get("vace_start_percent", 0.0)
        vace_end_percent = image_embeds.get("vace_end_percent", 1.0)
        vace_seqlen = image_embeds.get("vace_seq_len", None)

        vace_additional_embeds = image_embeds.get("additional_vace_inputs", [])
        if vace_context is not None:
            vace_data = [
                {"context": vace_context, 
                    "scale": vace_scale, 
                    "start": vace_start_percent, 
                    "end": vace_end_percent,
                    "seq_len": vace_seqlen
                    }
            ]
            if len(vace_additional_embeds) > 0:
                for i in range(len(vace_additional_embeds)):
                    if vace_additional_embeds[i].get("has_ref", False):
                        has_ref = True
                    vace_data.append({
                        "context": vace_additional_embeds[i]["vace_context"],
                        "scale": vace_additional_embeds[i]["vace_scale"],
                        "start": vace_additional_embeds[i]["vace_start_percent"],
                        "end": vace_additional_embeds[i]["vace_end_percent"],
                        "seq_len": vace_additional_embeds[i]["vace_seq_len"]
                    })

        noise = torch.randn(
                target_shape[0],
                target_shape[1] + 1 if has_ref else target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=seed_g)
        
        noise = noise.to(device, dtype)
        
        latent_video_length = noise.shape[1]  
        
        if samples is not None:
            input_samples = samples["samples"].squeeze(0).to(noise)
            if input_samples.shape[1] != noise.shape[1]:
                input_samples = torch.cat([input_samples[:, :1].repeat(1, noise.shape[1] - input_samples.shape[1], 1, 1), input_samples], dim=1)
            original_image = input_samples.to(device)
            if denoise_strength < 1.0:
                latent_timestep = timesteps[:1].to(noise)
                noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * input_samples

            mask = samples.get("mask", None)
            if mask is not None:
                if mask.shape[2] != noise.shape[1]:
                    mask = torch.cat([torch.zeros(1, noise.shape[0], noise.shape[1] - mask.shape[2], noise.shape[2], noise.shape[3]), mask], dim=2)

        init_latents = noise.to(device)
        
        fps_embeds = None
        if hasattr(transformer, "fps_embedding"):
            fps = round(fps, 2)
            log.info(f"Model has fps embedding, using {fps} fps")
            fps_embeds = [fps]
            fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]

        prefix_video = prefix_samples["samples"].to(noise) if prefix_samples is not None else None
        prefix_video_latent_length = prefix_video.shape[2] if prefix_video is not None else 0
        if prefix_video is not None:
            log.info(f"Prefix video of length: {prefix_video_latent_length}")
            init_latents[:, :prefix_video_latent_length] = prefix_video[0]
        
        disable_enhance() #not sure if this can work, disabling for now to avoid errors if it's enabled by another sampler

        freqs = None
        transformer.rope_embedder.k = None
        transformer.rope_embedder.num_frames = None
        if rope_function=="comfy":
            transformer.rope_embedder.k = 0
            transformer.rope_embedder.num_frames = latent_video_length
        else:
            d = transformer.dim // transformer.num_heads
            freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=0),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1)

        seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])
        log.info(f"Seq len: {seq_len}")
        seq_len = latent_video_length * 1560
        log.info(f"Seq len: {seq_len}")

        if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]: #default for latent2rgb
            from latent_preview import prepare_callback
        else:
            from ..latent_preview import prepare_callback #custom for tiny VAE previews
        

        #blockswap init        
        transformer_options = patcher.model_options.get("transformer_options", None)
        if transformer_options is not None:
            block_swap_args = transformer_options.get("block_swap_args", None)

        if block_swap_args is not None:
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", True)
            for name, param in transformer.named_parameters():
                if "block" not in name:
                    param.data = param.data.to(device)
                elif block_swap_args["offload_txt_emb"] and "txt_emb" in name:
                    param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)
                elif block_swap_args["offload_img_emb"] and "img_emb" in name:
                    param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)

            transformer.block_swap(
                block_swap_args["blocks_to_swap"] - 1 ,
                block_swap_args["offload_txt_emb"],
                block_swap_args["offload_img_emb"],
                vace_blocks_to_swap = block_swap_args.get("vace_blocks_to_swap", None),
            )

        elif model["auto_cpu_offload"]:
            for module in transformer.modules():
                if hasattr(module, "offload"):
                    module.offload()
                if hasattr(module, "onload"):
                    module.onload()
        elif model["manual_offloading"]:
            transformer.to(device)

        use_fresca = False
        if experimental_args is not None:
            video_attention_split_steps = experimental_args.get("video_attention_split_steps", [])
            if video_attention_split_steps:
                transformer.video_attention_split_steps = [int(x.strip()) for x in video_attention_split_steps.split(",")]
            else:
                transformer.video_attention_split_steps = []
            use_zero_init = experimental_args.get("use_zero_init", True)
            use_cfg_zero_star = experimental_args.get("cfg_zero_star", False)
            zero_star_steps = experimental_args.get("zero_star_steps", 0)

            use_fresca = experimental_args.get("use_fresca", False)
            if use_fresca:
                fresca_scale_low = experimental_args.get("fresca_scale_low", 1.0)
                fresca_scale_high = experimental_args.get("fresca_scale_high", 1.25)
                fresca_freq_cutoff = experimental_args.get("fresca_freq_cutoff", 20)

        #region model pred
        def model_pred(z, positive_embeds, negative_embeds, timestep, idx, image_cond=None, clip_fea=None, 
                             vace_data=None, unianim_data=None, teacache_state=None, kv_cache=None, crossattn_cache=None, current_kv_cache_start=0, kv_start=0, kv_end=0):
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=("fp8" in model["quantization"])):

                nonlocal patcher
                current_step_percentage = idx / len(timesteps)
                control_lora_enabled = False
                
                image_cond_input = image_cond
    
                base_params = {
                    'seq_len': seq_len,
                    'device': device,
                    'freqs': freqs,
                    't': timestep,
                    'current_step': idx,
                    'control_lora_enabled': control_lora_enabled,
                    'vace_data': vace_data,
                    'unianim_data': unianim_data,
                    'kv_cache': kv_cache,
                    'crossattn_cache': crossattn_cache,
                    'current_kv_cache_start': current_kv_cache_start,
                    "kv_start": kv_start, 
                    "kv_end": kv_end
                }

                #cond
                noise_pred_cond, teacache_state_cond = transformer(
                    [z], context=positive_embeds, y=[image_cond_input] if image_cond_input is not None else None,
                    clip_fea=clip_fea, is_uncond=False, current_step_percentage=current_step_percentage,
                    pred_id=teacache_state[0] if teacache_state else None,
                    **base_params
                )
                noise_pred_cond = noise_pred_cond[0].to(intermediate_device)
                
                if use_fresca:
                    noise_pred_cond = fourier_filter(
                        noise_pred_cond,
                        scale_low=fresca_scale_low,
                        scale_high=fresca_scale_high,
                        freq_cutoff=fresca_freq_cutoff,
                    )
                return noise_pred_cond, [teacache_state_cond]
            

        def convert_flow_pred_to_x0(flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            """
            Convert flow matching's prediction to x0 prediction.
            flow_pred: the prediction with shape [B, C, H, W]
            xt: the input noisy data with shape [B, C, H, W]
            timestep: the timestep with shape [B]

            pred = noise - x0
            x_t = (1-sigma_t) * x0 + sigma_t * noise
            we have x0 = x_t - sigma_t * pred
            see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
            """
            # use higher precision for calculations
            original_dtype = flow_pred.dtype
            flow_pred, xt, sigmas, timesteps = map(
                lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                            sample_scheduler.sigmas,
                                                            sample_scheduler.timesteps]
            )

            timestep_id = torch.argmin(
                (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
            sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
            x0_pred = xt - sigma_t * flow_pred
            return x0_pred.to(original_dtype)

        log.info(f"Sampling {(latent_video_length-1) * 4 + 1} frames at {init_latents.shape[3]*8}x{init_latents.shape[2]*8} with {steps} steps")

        intermediate_device = device

        #clear memory before sampling
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        #main loop
        self.num_frame_per_block = 3
        num_frames = noise.shape[1]
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        print("num_blocks: ", num_blocks)
        context_noise = 0
        self.frame_seq_length = 1560
        print("frame_seq_length: ", self.frame_seq_length)

        self.cache_window_size = self.frame_seq_length * num_frames
        self.shift_blocks = 1
        self.kv_start = 0
        self.kv_end = 0


        output_latents = torch.zeros(
            (target_shape[0],
                target_shape[1], 
             target_shape[2], 
             target_shape[3]), device=device, dtype=dtype)
        print("output_latents shape: ", output_latents.shape)

        # Step 1: Initialize KV cache to all zeros
        
        self._initialize_kv_cache(
            batch_size=1,
            dtype=noise.dtype,
            device=cache_device,
            num_blocks=transformer.num_layers,
            num_heads=transformer.num_heads,
        )
        self._initialize_crossattn_cache(
            batch_size=1,
            dtype=noise.dtype,
            device=cache_device,
            num_blocks=transformer.num_layers,
            num_heads=transformer.num_heads,
        )
                
        # Step 2: Cache context feature
        current_kv_cache_start_frame = 0
        num_input_frames = 0
        

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        print("all_num_frames", all_num_frames)

        pbar = ProgressBar(num_blocks)
        callback = prepare_callback(patcher, num_blocks)
        
        for i,current_num_frames in enumerate(all_num_frames):
            print("current_kv_cache_start_frame: ", current_kv_cache_start_frame)
            #noisy_input = noise[:, current_kv_cache_start_frame - num_input_frames:current_kv_cache_start_frame + current_num_frames - num_input_frames]
            noisy_input = noise[:, i * self.num_frame_per_block:(i + 1) * self.num_frame_per_block]
            print("noisy_input shape: ", noisy_input.shape)

            kv_end = self.kv_start + self.num_frame_per_block * self.frame_seq_length
            print("kv_end: ", kv_end)

            # Spatial denoising loop
            for step_index, current_timestep in enumerate(timesteps):
                print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [1, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if step_index < len(timesteps) - 1:
                    flow_pred, self.teacache_state = model_pred(
                        noisy_input.to(dtype), 
                        text_embeds["prompt_embeds"], 
                        text_embeds["negative_prompt_embeds"], 
                        timestep, step_index, image_cond, clip_fea, 
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_kv_cache_start=current_kv_cache_start_frame * self.frame_seq_length,
                        kv_start = self.kv_start,
                        kv_end = kv_end
                        )
                    
                    #print("noise_pred shape: ", noise_pred.shape)                    

                    denoised_pred = convert_flow_pred_to_x0(
                        flow_pred=flow_pred.transpose(0, 1),
                        xt=noisy_input.transpose(0, 1),
                        timestep=timestep.flatten(0, 1)
                    )
                    
                    next_timestep = timesteps[step_index + 1]
                    print("step_index: ", step_index, "next_timestep: ", next_timestep)
                    noisy_input = sample_scheduler.add_noise(
                        denoised_pred,
                        torch.randn_like(denoised_pred),
                        next_timestep * torch.ones(
                            [current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = noisy_input.transpose(0, 1)
                else:
                    # for getting real output
                    flow_pred, self.teacache_state = model_pred(
                        noisy_input.to(dtype), 
                        text_embeds["prompt_embeds"], 
                        text_embeds["negative_prompt_embeds"], 
                        timestep, step_index, image_cond, clip_fea, 
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_kv_cache_start=current_kv_cache_start_frame * self.frame_seq_length,
                        kv_start = self.kv_start,
                        kv_end = kv_end
                        )
                    
                    denoised_pred = convert_flow_pred_to_x0(
                        flow_pred=flow_pred.transpose(0, 1),
                        xt=noisy_input.transpose(0, 1),
                        timestep=timestep.flatten(0, 1)
                    )
                    denoised_pred = denoised_pred.transpose(0, 1)


            # Step 3.2: record the model's output
            #print("denoised_pred shape before output: ", denoised_pred.shape)
            #output_latents[:, current_kv_cache_start_frame:current_kv_cache_start_frame + current_num_frames] = denoised_pred
            output_latents[:, i * self.num_frame_per_block:(i + 1) * self.num_frame_per_block] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            
            print("cleaning KV cache")
            context_timestep = torch.ones_like(timestep) * context_noise
            model_pred(
                denoised_pred.to(dtype), 
                text_embeds["prompt_embeds"], 
                text_embeds["negative_prompt_embeds"], 
                context_timestep, step_index, image_cond, clip_fea, 
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_kv_cache_start=current_kv_cache_start_frame * self.frame_seq_length,
                kv_start = self.kv_start,
                kv_end = kv_end
                )
            
            # Update positions for next block
            r_shift_length = self.num_frame_per_block * self.frame_seq_length
            self.kv_start += r_shift_length
            kv_end += r_shift_length
            #self.rope_start += r_shift_length
            # Check if we need to shift the cache
            
            if kv_end > self.cache_window_size:
                print("Shifting KV cache")
                kv_end -= self._shift_kv_cache()

            # Step 3.4: update the start and end frame indices
            current_kv_cache_start_frame += current_num_frames


            if callback is not None:
                #callback_latent = output_latents[:, :current_kv_cache_start_frame].float().detach().permute(1,0,2,3)
                callback_latent = denoised_pred.float().detach().permute(1,0,2,3)
                callback(i, callback_latent, None, num_blocks)
            else:
                pbar.update(1)

        # reset cross attn cache
        for block_index in range(transformer.num_layers):
            self.crossattn_cache[block_index]["is_init"] = False
        # reset kv cache
        for block_index in range(len(self.kv_cache1)):
            self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=noise.device)
            self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=noise.device)
            
        self.kv_cache1 = None
        self.crossattn_cache = None
            
        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        try:
            print_memory(device)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        return ({
            "samples": output_latents.unsqueeze(0).cpu(),
            }, )

NODE_CLASS_MAPPINGS = {
    "WanVideoCausVidSampler": WanVideoCausVidSampler,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoCausVidSampler": "WanVideo CausVid Sampler",
    }
