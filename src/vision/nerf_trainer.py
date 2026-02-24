import os
from typing import Tuple, Union


from torch.optim import Optimizer
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm.notebook import tqdm
from IPython.display import display, clear_output

import src.vision.dataloader as dataloader
from src.vision.nerf_model_logic import NerfModel, render_image_nerf, get_rays, render_rays_batched, render_full_image_chunked


def positional_encoding(x, num_frequencies=6, incl_input=True):
  """Apply positional encoding to the input.

  Args:
    x (torch.Tensor): Input tensor to be positionally encoded. 
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding default: 6).
    incl_input (optional, bool): If True, concatenate the input with the 
        computed positional encoding (default: True).

    The list "results" will hold all the positional encodings. Appen
  
  Returns:
    (torch.Tensor): Positional encoding of the input tensor. 

  Example:
      >>> x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
      >>> encoded = positional_encoding(x, num_frequencies=4, incl_input=True)
      >>> print(encoded.shape)  # Example: [N, D * (2 * num_frequencies + 1)]

  Notes:
      - The section marked between `1(a) BEGIN` and `1(a) END` applies the sine
        and cosine transformations to the input tensor `x`, appending each 
        transformed tensor to a list of results.
      - Frequencies are scaled as powers of 2: `2^i * π` for `i` in 
        range(num_frequencies).

  """
  results = []
  if incl_input:
    results.append(x)
  #############################  1(a) BEGIN  ############################
  freq_band = 2 ** torch.arange(num_frequencies, dtype=x.dtype, device=x.device) * torch.pi
  xbroad = x.unsqueeze(1) * freq_band.unsqueeze(-1)

  sin = torch.sin(xbroad)
  cos = torch.cos(xbroad)

  encoded = torch.stack((sin, cos), dim=2)
  encoded = encoded.reshape(x.shape[0], -1, x.shape[-1])
  encoded = encoded.reshape(x.shape[0], -1)
  results.append(encoded)


# =============================NOT VECTORIZED SOLUTION: ===============================
  # nonVecres = []
  # if incl_input:
  #   nonVecres.append(x)
  # stackedFreq = []
  # for j, point in enumerate(x):
  #   encodedTensor = []
  #   for i in range(num_frequencies):
  #     resTensor = (pow(2, i) * torch.pi) * point
  #     encodedTensor.append(torch.sin(resTensor))
  #     encodedTensor.append(torch.cos(resTensor))
  #   grouped = torch.cat(encodedTensor)
  #   stackedFreq.append(grouped)

  #   # print(grouped.unsqueeze().shape)
  # # x[j] = torch.cat((point, grouped))
  # stackedFreq = torch.stack(stackedFreq)
  # # print(stackedFreq.shape, x.shape)
  # # print(stackedFreq)
  
  # # print(results)
  # nonVecres.append(stackedFreq)
  # # print(torch.cat(results, dim=1))

  # print(torch.cat(results, dim=-1) - torch.cat(nonVecres, dim=-1))
    
  # # # encode input tensor and append the encoded tensor to the list of results.
  # # raise NotImplementedError("You need to complete the codes in class positional_encoding!")
  
  #############################  1(a) END  ##############################
  return torch.cat(results, dim=-1)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    

class Trainer:

    #data_dir = "./nerf_synthetic/lego/"
    def __init__(self,
        data_dir: str,
        model_dir: str,
        model,
        optimizer,
        device,
        near_thresh,
        far_thresh,
        testpose,
        testimg,
        batch_size: int = 1,
        rays_per_batch: int = 2048,
        load_from_disk: bool = True,
        cuda: bool = False,
        num_frequencies=(6, 4),
        depth_samples_per_ray=64, 
        lr=5e-4, num_iters=1000,
        display_every=25,
        seed=4476,
        scaled_size : tuple[int, int] = (100, 100)
    ) -> None:
        self.data_dir = data_dir
        
        self.device = device
        self.batch_size = batch_size
        self.rays_per_batch = rays_per_batch
        self.load_from_disk = load_from_disk
        self.cuda = cuda
        self.near_thresh = near_thresh
        self.far_thresh = far_thresh
        self.depth_samples_per_ray = depth_samples_per_ray

        self.testpose = testpose.to(self.device)
        self.testimg = testimg.to(self.device)
        dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}

        self.train_dataset = dataloader.SpatialImg(
            self.data_dir, split="train", scaled_size=scaled_size
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, **dataloader_args
        )


        self.test_dataset = dataloader.SpatialImg(
            self.data_dir, split="test", scaled_size=scaled_size
        )
        self.test_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, **dataloader_args
        )


        self.intrinsics = self.train_dataset.cam_intrinsics.to(device)

        self.encode_pts = lambda x: positional_encoding(x, num_frequencies=num_frequencies[0])
        self.encode_view= lambda x: positional_encoding(x, num_frequencies=num_frequencies[1])
        self.encode_channels = num_frequencies[0] * 2 * 3 + 3


        self.model = model
        self.optimizer = optimizer
        # self.model = NerfModel(self.encode_channels, filter_size=256, freq=num_frequencies)
        # self.model.to(device)
        # self.model.apply(self.weights_init)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.h, self.w = self.train_dataset.get_hw()

        self.iteration_numbers = []

        self.training_psnr_history = []
        self.validation_psnr_history = []

        self.train_snap_rgb = []
        self.train_snap_depth = []
        self.train_loss_history = []
        self.validation_loss_history = []
        

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        os.makedirs('output', exist_ok=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model.train()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    

    def run_training_loop(self, num_epochs: int, show_every: int, batched: bool=True):
        plt.ioff()
        fig, axes = plt.subplots(1, 5, figsize=(20,5))
        # self.model.train()
        for epoch_idx in tqdm(range(num_epochs)):
            tqdm.write(f"Starting Epoch {epoch_idx}")
            # train_loss, train_psnr = self.train_epoch(fig, axes, epoch_idx, show_every)
            if batched:
                self.train_batched_epoch(fig, axes, epoch_idx, show_every)
            else:
                self.train_epoch(fig, axes, epoch_idx, show_every)
            time.sleep(0.01)

        print("Training completed!")
        # plt.ioff()
        # plt.show()
        self.train_snap_rgb_tensor = np.stack(self.train_snap_rgb)
        self.train_snap_depth_tensor = np.stack(self.train_snap_depth)

        print(self.train_snap_rgb_tensor.shape, self.train_snap_depth_tensor.shape)
        return self.model, self.encode_pts, self.encode_view, self.train_snap_rgb_tensor, self.train_snap_depth_tensor

    def train_batched_epoch(self, fig, axes, epoch_idx, show_every:int) -> Tuple[float, float]:
        self.model.train()
        for i, (target_img, target_tform_cam2world) in enumerate(tqdm(self.train_loader, desc="ProcessingBatches")):
            
            target_img = target_img.to(self.device).squeeze(0)
            target_tform_cam2world= target_tform_cam2world.to(self.device).squeeze(0)

            rays_o, rays_d = get_rays(self.h, self.w, self.intrinsics, target_tform_cam2world)

            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            target_img_flat = target_img.reshape(-1, 3)

            num_rays = rays_o.shape[0]
            select_idxs = np.random.choice(num_rays, size=[self.rays_per_batch], replace=False)

            rays_o_batched = rays_o[select_idxs]
            rays_d_batched = rays_d[select_idxs]
            target_rgb_batch = target_img_flat[select_idxs]
            
            rgb_rendered, depth_pred = render_rays_batched(rays_o_batched, rays_d_batched, target_tform_cam2world, 
                                                   self.near_thresh, self.far_thresh, self.depth_samples_per_ray, 
                                                   self.encode_pts, self.encode_view , self.model, True)
            
            # rgb_rendered, depth_pred = render_full_image_chunked(self.h, self.w, self.intrinsics, target_tform_cam2world, 
            #                                        self.near_thresh, self.far_thresh, self.depth_samples_per_ray, 
            #                                        self.encode_pts, self.encode_view , self.model, True)
            
            # print(rgb_rendered.shape, target_rgb_batch.shape)
            # loss = F.mse_loss(rgb_rendered, target_img)
            loss = F.mse_loss(rgb_rendered, target_rgb_batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.train_loss_history.append(loss.item())


            # torch.cuda.empty_cache()
            # for ax in axes: 
            #     ax.clear()
            # axes[0].imshow(rgb_rendered.detach().cpu().numpy())
            # axes[1].imshow(target_img.detach().cpu().numpy()); axes[1].set_title("Target")
            # axes[2].plot(np.array(self.train_loss_history)); axes[2].set_title("Loss")
            # axes[3].imshow(depth_pred.detach().cpu().numpy()); axes[3].set_title("Depth")
            # display(fig)
            # clear_output(wait=True)

            if i % show_every == 0:
                print("image pixels: ", rays_o.shape, " | rays selected: ", rays_o_batched.shape)
                torch.cuda.empty_cache()
                self.validate_and_plot(fig, axes, i + (epoch_idx * len(self.train_loader)), batched=True)


    def train_epoch(self, fig, axes, epoch_idx, show_every: int) -> Tuple[float, float]:
        
        for i, (target_img, target_tform_cam2world) in enumerate(tqdm(self.train_loader, desc="Processing Batches")):
            target_img = target_img.to(self.device).squeeze(0)
            target_tform_cam2world= target_tform_cam2world.to(self.device).squeeze(0)

            # print("t_img: ", target_img.shape, " t_pose: ", target_tform_cam2world.shape)

            
            rgb_rendered, depth_pred = render_image_nerf(self.h, self.w, self.intrinsics, target_tform_cam2world, 
                                                   self.near_thresh, self.far_thresh, self.depth_samples_per_ray, 
                                                   self.encode_pts, self.encode_view , self.model, True)

            # plt.imshow(rgb_rendered.detach().cpu().numpy())
            # plt.imshow(target_img.detach().cpu().numpy())

            

            loss = F.mse_loss(rgb_rendered, target_img)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


            torch.cuda.empty_cache()
            for ax in axes: 
                ax.clear()
            axes[0].imshow(rgb_rendered.detach().cpu().numpy())
            axes[1].imshow(target_img.detach().cpu().numpy()); axes[1].set_title("Target")
            axes[2].plot(np.array(self.train_loss_history)); axes[2].set_title("Loss")
            axes[3].imshow(depth_pred.detach().cpu().numpy()); axes[3].set_title("Depth")
            display(fig)
            clear_output(wait=True)

            if i % show_every == 0:
                torch.cuda.empty_cache()
                self.validate_and_plot(fig, axes, i + (epoch_idx * len(self.train_loader)))


            self.train_loss_history.append(loss.item())
            # psnr = -10. * torch.log10(loss)
            # self.training_psnr_history.append(psnr)
            # self.plot_psnr(fig, axes, epoch_idx, renderargs)

    def validate_and_plot(self, fig, axes, iternums, batched=False):
        # self.model.eval()
        with torch.no_grad():
            if batched:
                # self.model.eval()
                rgb_pred, depth_pred = render_full_image_chunked(self.h, self.w, self.intrinsics, self.testpose, 
                                                   self.near_thresh, self.far_thresh, self.depth_samples_per_ray, 
                                                   self.encode_pts, self.encode_view , self.model, True)
                
            else:
                rgb_pred, depth_pred = render_image_nerf(self.h, self.w, self.intrinsics, self.testpose, 
                                                   self.near_thresh, self.far_thresh, self.depth_samples_per_ray, 
                                                   self.encode_pts, self.encode_view , self.model, True)


            
            loss = F.mse_loss(rgb_pred, self.testimg)
            psnr = -10. * torch.log10(loss)
            self.iteration_numbers.append(iternums) 
            self.training_psnr_history.append(psnr.item())
            self.train_snap_rgb.append(rgb_pred.detach().cpu().numpy())
            self.train_snap_depth.append(depth_pred.detach().cpu().numpy())
            # print(len(self.iteration_numbers), len(self.training_psnr_history))

        for ax in axes: 
            ax.clear()
        axes[0].imshow(rgb_pred.cpu().numpy()); axes[0].set_title(f"Iter {iternums}")
        axes[1].imshow(self.testimg.cpu().numpy()); axes[1].set_title("Target")
        axes[2].imshow(depth_pred.cpu().numpy()); axes[2].set_title("Depth")
        axes[3].plot(self.iteration_numbers, self.training_psnr_history); axes[3].set_title("PSNR")
        axes[4].plot(np.array(self.train_loss_history)); axes[4].set_title("Loss")
        
        
        # self.model.train()
        display(fig)
        clear_output(wait=True)