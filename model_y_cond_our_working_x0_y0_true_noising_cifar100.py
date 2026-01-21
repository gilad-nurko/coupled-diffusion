import torch.nn as nn
import torch
from torchvision.transforms import GaussianBlur
import math
from tqdm import tqdm


LOGITS_MEAN = 0.0
LOGITS_STD = 0.8 # based on stats on our classifier logits


class MNISTDiffusion(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8],classifier=None,is_cond=False,is_y_cond=False,guiding_noise_level=0.15,is_ddim=True,num_classes=100,mode="cifar100",device="cuda",pretrained_model_y_ckpt=None,corruption_type='gaussian_blur',blur_sigma=2.0):
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size
        self.guiding_noise_level=guiding_noise_level
        self.is_y_cond=is_y_cond
        self.is_ddim=is_ddim
        self.mode = mode
        self.corruption_type = corruption_type
        if self.corruption_type == 'gaussian_blur':
            # Hardcoded kernel_size=5
            self.blur_transform = GaussianBlur(kernel_size=5, sigma=blur_sigma)

        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        if self.mode == "cifar100":
            mean = torch.tensor((0.5071, 0.4867, 0.4408)).view(1,3,1,1)
            std  = torch.tensor((0.2675, 0.2565, 0.2761)).view(1,3,1,1)
        else:
            mean = torch.tensor((0.4811, 0.4575, 0.4079)).view(1,3,1,1)
            std  = torch.tensor((0.2604, 0.2532, 0.2682)).view(1,3,1,1)
        self.register_buffer("data_mean", mean)
        self.register_buffer("data_std", std)

        if mode=="cifar10":
            from y_probs_model_cifar10 import ConditionalModel
            self.model_y=ConditionalModel(n_input_channels=in_channels,num_classes=num_classes)
        if mode=="mnist":
            from y_probs_model import ConditionalModel
            self.model_y=ConditionalModel(n_input_channels=in_channels,num_classes=num_classes)
        if mode=="cifar100" or mode=="imagenet32" or mode=="imagenet32_1k":
            if mode=="imagenet32_1k":
                from y_probs_model_imagenet_1k import ConditionalModel
            else:
                from y_probs_model_cifar100 import ConditionalModel
            self.model_y = ConditionalModel(
                feature_dim=512,
                hidden_dim=1024,
                n_input_channels=in_channels,   # 3
                num_classes=num_classes,        # 100
                timesteps=150,
                num_heads=8,
            )
            if pretrained_model_y_ckpt is not None:
                ckpt = torch.load(pretrained_model_y_ckpt, map_location=device)

                if "logits_model_state_dict" in ckpt:
                    state_dict = ckpt["logits_model_state_dict"]
                else:
                    # fallback: extract from diffusion_state_dict if needed
                    sd = ckpt["diffusion_state_dict"]
                    state_dict = {
                        k.replace("model.", ""): v
                        for k, v in sd.items()
                        if k.startswith("model.")
                    }

                missing, unexpected = self.model_y.load_state_dict(state_dict, strict=False)
                print(f"[model_y] loaded pretrained weights from {pretrained_model_y_ckpt}")
                print(f"  missing keys   : {len(missing)}")
                print(f"  unexpected keys: {len(unexpected)}")
                
        if mode=="imagenet32_1k":
            from unet_y_cond_1000_classes import Unet
        else:
            from unet_y_cond import Unet

        self.model_x=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults,is_cond=is_cond,is_y_cond=is_y_cond,num_classes=num_classes)
        self.classifier=classifier
        self.num_classes=num_classes
    
    def _normalize_image(self, x):
        return (x - self.data_mean) / self.data_std


    def forward(self,x,noise,is_x=True,target=None,cond=None,t=None):
        # x:NCHW
        signal_0 = None
        def get_guiding_cond(input_img):
            if self.corruption_type == 'pixel_noise':
                noise_cond = torch.randn_like(input_img).to(input_img.device)
                mask = torch.bernoulli(torch.full(input_img.shape, self.guiding_noise_level, device=input_img.device))
                return input_img * (1 - mask) + noise_cond * mask
            elif self.corruption_type == 'gaussian_blur':
                return self.blur_transform(input_img)
            return input_img
        if is_x:
            x_t=self._forward_diffusion(x,t,noise)
            guiding_cond = get_guiding_cond(x)
            y_cond = cond 
            pred_noise = self.model_x(x_t, t, guiding_cond, y_cond=y_cond)
            alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)            
            x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred_noise
            x_0_pred.clamp_(-1., 1.)
            x_0_pred=(x_0_pred+1.)/2 #[-1,1] to [0,1]
            x_0_pred = self._normalize_image(x_0_pred) # added for cifar100
            signal_0 = x_0_pred
        else:
            with torch.no_grad():
                guiding_cond = get_guiding_cond(x)
                classifier_input = torch.where(t[:, None, None, None] < self.timesteps*0.5, cond, guiding_cond) # changed from 0.2
                class_logits = self.classifier(classifier_input)
                # temprature = 3.0
                # class_logits = class_logits / temprature
                # class_probs = torch.softmax(class_logits, dim=1)
                # y_cond = class_probs
                # normalize class_logits to have mean and std of LOGITS_MEAN and LOGITS_STD
                class_mean = class_logits.mean(dim=1, keepdim=True)
                class_std = class_logits.std(dim=1, keepdim=True)
                class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
                y_cond = class_logits
                # noise_probs = torch.softmax(noise,dim=1)
                y_t=self._forward_diffusion(target,t,noise,is_x=False)
            
            t_norm = t.float() / (self.timesteps - 1)
            pred_noise = self.model_y(y_t, t_norm, y_cond=y_cond, x=classifier_input)
            alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(y_t.shape[0],1)            
            y_0_pred=torch.sqrt(1. / alpha_t_cumprod)*y_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred_noise
            # normalize y_0_pred to have mean and std of LOGITS_MEAN and LOGITS_STD
            y_0_mean = y_0_pred.mean(dim=1, keepdim=True)
            y_0_std = y_0_pred.std(dim=1, keepdim=True)
            y_0_pred = LOGITS_STD * (y_0_pred - y_0_mean) / (y_0_std + 1e-5) + LOGITS_MEAN
            signal_0 = y_0_pred
        return pred_noise, signal_0

    # @torch.no_grad()
    # def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda",x_cond=None):
    #     x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
    #     y_t = torch.randn((n_samples, self.num_classes)).to(device)
    #     # y_t=torch.softmax(y_t,dim=1)  # TODO:maybe remove this line
    #     y_cond=None
    #     with torch.no_grad():
    #         class_logits = self.classifier(x_cond)
    #         # temprature = 3.0
    #         # class_logits = class_logits / temprature
    #         # class_probs = torch.softmax(class_logits, dim=1)
    #         # y_cond = class_probs
    #         # normalize class_logits to have mean and std of LOGITS_MEAN and LOGITS_STD
    #         class_mean = class_logits.mean(dim=1, keepdim=True)
    #         class_std = class_logits.std(dim=1, keepdim=True)
    #         class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
    #         y_cond = class_logits
            
    #     y_image_cond=x_cond
    #     for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
    #         noise_x=torch.randn_like(x_t).to(device)
    #         noise_y=torch.randn_like(y_t).to(device)
    #         t=torch.tensor([i for _ in range(n_samples)]).to(device)
    #         if i<self.timesteps*0.5: # rely on x_t rather then the guiding, changed from 0.5
    #             y_image_cond=x_0_pred
    #             with torch.no_grad():
    #                 class_logits = self.classifier(x_0_pred)
    #                 # temprature = 3.0
    #                 # class_logits = class_logits / temprature
    #                 # class_probs = torch.softmax(class_logits, dim=1)
    #                 # y_cond = class_probs
    #                 # normalize class_logits to have mean and std of LOGITS_MEAN and LOGITS_STD
    #                 class_mean = class_logits.mean(dim=1, keepdim=True)
    #                 class_std = class_logits.std(dim=1, keepdim=True)
    #                 class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
    #                 y_cond = class_logits

    #         if clipped_reverse_diffusion:
    #             y_t, y_0_pred=self._reverse_diffusion_with_clip(y_t,t,noise_y,y_image_cond,y_cond,is_x=False)
    #             x_t, x_0_pred=self._reverse_diffusion_with_clip(x_t,t,noise_x,x_cond,y_0_pred,is_x=True)
    #         else:
    #             y_t, y_0_pred=self._reverse_diffusion(y_t,t,noise_y,y_image_cond,y_cond,is_x=False)
    #             x_t, x_0_pred=self._reverse_diffusion(x_t,t,noise_x,x_cond,y_0_pred,is_x=True)

    #     x_t=(x_t+1.)/2. #[-1,1] to [0,1]
    #     x_t = self._normalize_image(x_t)  # added for cifar100
    #     # y_t=(y_t+1.)/2. #[-1,1] to [0,1]    # TODO:maybe remove this line
    #     # y_t=torch.softmax(y_t, dim=1)    # TODO:maybe remove this line

    #     # normalize y_t to have mean and std of LOGITS_MEAN and LOGITS_STD
    #     y_mean = y_t.mean(dim=1, keepdim=True)
    #     y_std = y_t.std(dim=1, keepdim=True)
    #     y_t = LOGITS_STD * (y_t - y_mean) / (y_std + 1e-5) + LOGITS_MEAN

    #     return x_t,y_t
    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cuda", x_cond=None, sampling_steps=None):
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device)
        y_t = torch.randn((n_samples, self.num_classes)).to(device)
        y_cond = None
        
        # Setup step indices
        S = self.timesteps if (sampling_steps is None) else int(sampling_steps)
        if S == self.timesteps:
            step_indices = torch.arange(self.timesteps - 1, -1, -1, device=device)
        else:
            step_indices = torch.linspace(self.timesteps - 1, 0, steps=S, device=device).long()

        with torch.no_grad():
            class_logits = self.classifier(x_cond)
            class_mean = class_logits.mean(dim=1, keepdim=True)
            class_std = class_logits.std(dim=1, keepdim=True)
            class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
            y_cond = class_logits
            
        y_image_cond = x_cond
        
        # Iterate over subsampled indices
        for i in tqdm(step_indices, desc=f"Sampling ({len(step_indices)} steps)"):
            noise_x = torch.randn_like(x_t).to(device)
            noise_y = torch.randn_like(y_t).to(device)
            
            # Use 'i' (which contains the actual timestep value, e.g., 149, 147...)
            val_i = int(i.item())
            t = torch.full((n_samples,), val_i, device=device, dtype=torch.long)
            
            # Logic check: val_i is the actual timestep value, so logic works even with subsampling
            if val_i < int(S * 0.5): 
                y_image_cond = x_0_pred
                with torch.no_grad():
                    class_logits = self.classifier(x_0_pred)
                    class_mean = class_logits.mean(dim=1, keepdim=True)
                    class_std = class_logits.std(dim=1, keepdim=True)
                    class_logits = LOGITS_STD * (class_logits - class_mean) / (class_std + 1e-5) + LOGITS_MEAN
                    y_cond = class_logits

            if clipped_reverse_diffusion:
                y_t, y_0_pred = self._reverse_diffusion_with_clip(y_t, t, noise_y, y_image_cond, y_cond, is_x=False)
                x_t, x_0_pred = self._reverse_diffusion_with_clip(x_t, t, noise_x, x_cond, y_0_pred, is_x=True)
            else:
                y_t, y_0_pred = self._reverse_diffusion(y_t, t, noise_y, y_image_cond, y_cond, is_x=False)
                x_t, x_0_pred = self._reverse_diffusion(x_t, t, noise_x, x_cond, y_0_pred, is_x=True)
        
        x_t = (x_t + 1.) / 2. 
        x_t = self._normalize_image(x_t)
        
        y_mean = y_t.mean(dim=1, keepdim=True)
        y_std = y_t.std(dim=1, keepdim=True)
        y_t = LOGITS_STD * (y_t - y_mean) / (y_std + 1e-5) + LOGITS_MEAN
        return x_t, y_t
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise,is_x=True):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        if is_x:
            return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise
        else:
            return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1)*noise


    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise,x_cond=None,y_cond=None,is_x=True):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        if is_x:
            pred=self.model_x(x_t,t,x_cond,y_cond)
            alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
            alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
            beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
            sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
            mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

            if t.min()>0:
                alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
                std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
            else:
                std=0.0

            return mean+std*noise
        else:
            t_norm = t.float() / (self.timesteps - 1)
            pred=self.model_y(x_t,t_norm,y_cond,x_cond)
            alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1)
            alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1)
            beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1)
            sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1)
            mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

            if t.min()>0:
                alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1)
                std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
            else:
                std=0.0

            if self.is_ddim:
                std=0.0

            return mean+std*noise 


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self,x_t,t,noise,x_cond=None,y_cond=None,is_x=True): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        if is_x:
            pred=self.model_x(x_t,t,x_cond,y_cond)
            alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
            alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
            beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
            
            x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
            x_0_pred.clamp_(-1., 1.)

            if t.min()>0:
                alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
                mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                    ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

                std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
            else:
                mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
                std=0.0

            x_0_pred=(x_0_pred+1.)/2 #[-1,1] to [0,1]
            x_0_pred = self._normalize_image(x_0_pred)
            return mean+std*noise, x_0_pred
        else:
            t_norm = t.float() / (self.timesteps - 1)
            pred=self.model_y(x_t,t_norm,y_cond,x_cond)
            alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1)
            alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1)
            beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1)
            
            x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
            # normalize x_0_pred to have mean and std of LOGITS_MEAN and LOGITS_STD
            x_0_mean = x_0_pred.mean(dim=1, keepdim=True)
            x_0_std = x_0_pred.std(dim=1, keepdim=True)
            x_0_pred = LOGITS_STD * (x_0_pred - x_0_mean) / (x_0_std + 1e-5) + LOGITS_MEAN
            # x_0_pred.clamp_(-1., 1.)

            if t.min()>0:
                alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1)
                mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                    ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

                std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
            else:
                mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
                std=0.0
                
            if self.is_ddim:
                std=0.0
            # x_0_pred=(x_0_pred+1.)/2 #[-1,1] to [0,1]
            # x_0_pred = torch.softmax(x_0_pred, dim=1)
            return mean+std*noise, x_0_pred
        