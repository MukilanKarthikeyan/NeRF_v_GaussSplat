import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable

class NerfModel(nn.Module):
    
    def __init__(self, in_channels: int, filter_size: int=256, freq: tuple[int, int]=(6,4)):
        """This network will have a total of 8 fully connected layers. The activation function will be ReLU

        The number of input features to layer 5 will be a bit different. Refer to the docstring for the forward pass.
        Do not include an activation after layer 8 in the Sequential block. Layer 8's should output 4 features.

        Args
        ---
        in_channels (int): the number of input features from 
            the data
        filter_size (int): the number of in/out features for all layers. Layers 1 (because of in_channels), 5, and 8 are
            a bit different.
        """
        super().__init__()

        self.fc_layers_group1: nn.Sequential = None  # For layers 1-3
        self.layer_4: nn.Linear = None
        self.fc_layers_group2: nn.Sequential = None  # For layers 5-8
        self.loss_criterion = None
        self.in_channels = in_channels
        self.view_encoding_size = 2 + (2*2*freq[1])

        ##########################################################################
        # Student code begins here
        ##########################################################################
        self.fc_layers_group1 = nn.Sequential(
            nn.Linear(in_channels, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU()
        )
        self.layer_4 = nn.Linear(filter_size, filter_size)


        self.fc_layers_group2 = nn.Sequential(
            nn.Linear(2*filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size + 1)
        )

        self.view_synth_layer = nn.Sequential(
            nn.Linear(filter_size + self.view_encoding_size, 128),
            nn.ReLU(),
            nn.Linear(128,3),
            # nn.Sigmoid()
        )


        # self.fc_layers_group2_view = nn.Sequential(
        #     nn.Linear(2*filter_size, filter_size),
        #     nn.ReLU(),
        #     nn.Linear(filter_size, filter_size),
        #     nn.ReLU(),
        #     nn.Linear(filter_size, filter_size),
        #     nn.ReLU(),
        #     nn.Linear(filter_size, 257)
        # )
        # self.fc_mlp_view_color = nn.Linear(256, 128)
        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.apply(self._weights_init)
        
        # raise NotImplementedError('`init` function in `NerfModel` needs to be implemented')

        ##########################################################################
        # Student code ends here
        ##########################################################################
  

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of the model. 
        
        NOTE: The input to layer 5 should be the concatenation of post-activation values from layer 4 with 
        post-activation values from layer 3. Therefore, be extra careful about how self.layer_4 is used, the order of
        concatenation, and what the specified input shape to layer 5 should be. The output from layer 5 and the 
        dimensions thereafter should be filter_size.
        
        Args
        ---
        x (torch.Tensor): input of shape 
            (batch_size, in_channels)
        
        Returns
        ---
        rgb (torch.Tensor): The predicted rgb values with 
            shape (batch_size, 3)
        sigma (torch.Tensor): The predicted density values with shape (batch_size)
        """
        rgb = None
        sigma = None

        ##########################################################################
        # Student code begins here
        ##########################################################################
        position = x[: ,:self.in_channels]
        view = x[:, self.in_channels:]
        x_third = self.fc_layers_group1(position)
        x_fourth = self.layer_4(x_third)
        x_fourth = F.relu(x_fourth)
        
        x_residual = torch.cat((x_fourth, x_third), dim=-1)
        precolor_output = self.fc_layers_group2(x_residual)
        sigma = precolor_output[:, -1]
        # print("shapes of cat: ", view.shape, precolor_output[:,:-1].shape)
        view_aug_logit = torch.cat((precolor_output[:,:-1], view), dim=-1)
        # print("view_aug shape: ", view_aug_logit.shape)
        rgb = self.view_synth_layer(view_aug_logit)
        # print(rgb.shape)

        # rgb = output[:, :3]
        

        rgb = F.sigmoid(rgb)
        sigma = F.relu(sigma)

# =========================View Dependent=============================
        # x_third = self.fc_layers_group1(x)
        # x_fourth = self.layer_4(x_third)
        # x_fourth = F.relu(x_fourth)
        
        # x_residual = torch.cat((x_fourth, x_third), dim=-1)
        # x_pos = self.fc_layers_group2(x_residual)

        # x_pos_dep = torch.cat(x_pos, )


        
        # sigma = output[:, -1]
        

        # rgb = output[:, :3]

        # raise NotImplementedError('`forward` function in `NerfModel` needs to be implemented')

        ##########################################################################
        # Student code ends here
        ##########################################################################

        return rgb, sigma
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def get_rays(height: int, width: int, intrinsics: torch.Tensor, tform_cam2world: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).
    
    Args
    ---
    height (int): 
        the height of an image.
    width (int): the width of an image.
    intrinsics (torch.Tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    
    Returns
    ---
    ray_origins (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the centers of
        each ray. Note that desipte that all ray share the same origin, 
        here we ask you to return the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the
        direction of each ray.
    """
    device = tform_cam2world.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

    ##########################################################################
    # Student code begins here
    ##########################################################################
    # cam_origin = -tform_cam2world[:3, :3].T * tform_cam2world[-1, :3]
    cam_origin = tform_cam2world[:3, 3]
    ray_origins = cam_origin.unsqueeze(0).unsqueeze(0).expand(height, width, 3)

    i_grid = torch.arange(height, device=device)
    j_grid = torch.arange(width, device=device)

    ii, jj = torch.meshgrid(i_grid, j_grid, indexing="ij")
    # print(ii, jj, height, width)
    ones = torch.ones_like(ii)
    # print([ii, jj, ones])
    ray_camera = torch.stack([jj, ii, ones], dim=-1).float()
    # print(ray_camera)

    ray_dir_camera = ray_camera @ torch.linalg.inv(intrinsics).T 
    # print(ray_dir_camera)
    #  = torch.zeros(4, 4)
    # cam_to_world[-1, :3] = cam_origin
    # cam_to_world[-1, -1] = 1
    rotation = tform_cam2world[:3, :3]

    ray_directions = ray_dir_camera @ rotation.T
    # print(ray_origins.shape, ray_directions.shape)
    # print(ray_directions)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    

    ##########################################################################
    # Student code ends here
    ##########################################################################

    return ray_origins, ray_directions

def sample_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize:bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 3D points on the given rays. The near_thresh and far_thresh
    variables indicate the bounds of sampling range.
    
    Args
    ---
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    query_points (torch.Tensor): Query 3D points along each ray
        (shape: :math:`(height, width, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    """
    device = ray_origins.device
    height, width = ray_origins.shape[:2]
    depth_values = torch.zeros((height, width, num_samples), device=device) # placeholder
    query_points = torch.zeros((height, width, num_samples, 3), device=device) # placeholder
    
    ##########################################################################
    # Student code begins here
    ##########################################################################
    bounds = (far_thresh - near_thresh)/num_samples

    i = torch.arange(num_samples, device=device)
    query_ts = near_thresh + (i * bounds)
    depth_values = query_ts.unsqueeze(0).unsqueeze(0).expand(height, width, num_samples)
    if randomize:
        random_samp = torch.rand(height, width, num_samples, device=device)
        depth_values = (bounds * random_samp) + depth_values


    query_points = ray_origins.unsqueeze(-2) + depth_values.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    # print(query_ts.shape, ray_directions.shape)
    # offsets = depth_values.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    # print(offsets, ray_origins)
    # query_points = ray_origins.unsqueeze(-2) + offsets
    # print(offsets.shape)

    # depth_values = torch.sum(((depth_values)**2), dim=-1)**0.5
        
        # [near_thresh + (i-1)/num_samples*(bounds) for i in range(num_samples)]
    # raise NotImplementedError('`sample_points_from_rays()` function needs to be implemented')

    ##########################################################################
    # Student code ends here
    ##########################################################################
    
    return query_points, depth_values


def sample_points_from_rays_general(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize:bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 3D points on the given rays. The near_thresh and far_thresh
    variables indicate the bounds of sampling range.
    
    Args
    ---
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(..., 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(..., 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    query_points (torch.Tensor): Query 3D points along each ray
        (shape: :math:`(height, width, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    """
    device = ray_origins.device
    shape_prefix = ray_origins.shape[:-1]
    
    ##########################################################################
    # Student code begins here
    ##########################################################################

    t_vals = torch.linspace(near_thresh, far_thresh, steps=num_samples+1, device=device)
    lower = t_vals[:-1]
    if randomize:        
        upper = t_vals[1:]
        noise = torch.rand((*shape_prefix, num_samples), device=device)
        depth_values = lower + (upper - lower)*noise
    else:
        depth_values = lower.expand((*shape_prefix, num_samples))


    # bounds = (far_thresh - near_thresh)/num_samples
     
    # i = torch.arange(num_samples, device=device)
    # query_ts = near_thresh + (i * bounds)
    # depth_values = query_ts.unsqueeze(0).unsqueeze(0).expand(*shape_prefix, num_samples)
    # if randomize:
    #     random_samp = torch.rand(*shape_prefix, num_samples, device=device)
    #     depth_values = (bounds * random_samp) + depth_values


    query_points = ray_origins.unsqueeze(-2) + depth_values.unsqueeze(-1) * ray_directions.unsqueeze(-2)
  

    ##########################################################################
    # Student code ends here
    ##########################################################################
    
    return query_points, depth_values

def get_viewing_direction(
        cam_ray_origin: torch.Tensor, 
        cam_ray_dir: torch.Tensor, 
        tform_cam2world: torch.Tensor) -> torch.Tensor:
    
    device = cam_ray_origin.device
    orig_shape = cam_ray_dir.shape

    cam_ray_dir = F.normalize(cam_ray_dir, p=2, dim=-1)


    R_cam2world = tform_cam2world[:3, :3]
    cam_ray_dir_flat = cam_ray_dir.reshape(-1, 3)


    # z_cam = torch.tensor([0, 0, 1], device=device).float()
    view_dir = cam_ray_dir_flat @ R_cam2world.T
    view_dir = view_dir.reshape(orig_shape)


    theta = torch.atan2(view_dir[..., 1], view_dir[..., 0])
    phi = torch.acos(torch.clamp(view_dir[..., 2], -1.0, 1.0))
    


    return torch.stack((theta, phi), dim=-1)


    # angles =  pytorch3d.transforms.rotation_matrix_to_euler_angles()
def cumprod_exclusive(x: torch.Tensor) -> torch.Tensor:
    """ Helper function that computes the cumulative product of the input tensor, excluding the current element
    Example:
    > cumprod_exclusive(torch.tensor([1,2,3,4,5]))
    > tensor([ 1,  1,  2,  6, 24])
    
    Args:
    -   x: Tensor of length N
    
    Returns:
    -   cumprod: Tensor of length N containing the cumulative product of the tensor
    """

    cumprod = torch.cumprod(x, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

def compute_compositing_weights(sigma: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """This function will compute the compositing weight for each query point.

    Args
    ---
    sigma (torch.Tensor): Volume density at each query location (X, Y, Z)
        (shape: :math:`(height, width, num_samples)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    
    Returns:
    weights (torch.Tensor): Rendered compositing weight of each sampled point 
        (shape: :math:`(height, width, num_samples)`).
    """

    device = depth_values.device
    weights = torch.ones_like(sigma, device=device) # placeholder

    ##########################################################################
    # Student code begins here
    ##########################################################################
    delta = torch.zeros_like(depth_values)
    delta[..., :-1] = depth_values[..., 1:] - depth_values[..., :-1]
    delta[...,-1] = 1e9
    
    
    sig_del = torch.clamp(sigma * delta, min=0.0, max=50.0)
    exp_sig_del = torch.exp(-sig_del)
    # sig_del = torch.exp(-sigma * delta)
    transmittance = cumprod_exclusive(exp_sig_del)
    
    # for i in range(len(depth_values)):
    #     transmittance[i] = torch.exp(-torch.sum(sig_del[:i]))

    alpha = 1 - exp_sig_del
    weights = transmittance * alpha
    # raise NotImplementedError('`compute_compositing_weights()` function needs to be implemented')

    ##########################################################################
    # Student code ends here
    ##########################################################################

    return weights

def get_minibatches(inputs: torch.Tensor, chunksize: int = 1024 * 32) -> list[torch.Tensor]:
    """Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def render_image_nerf(height: int, width: int, intrinsics: torch.Tensor, tform_cam2world: torch.Tensor,
                      near_thresh: float, far_thresh: float, depth_samples_per_ray: int,
                      encoding_function_pts: Callable, encoding_function_view: Callable, model:NerfModel, rand:bool=False) \
                      -> tuple[torch.Tensor, torch.Tensor]:
    """ This function will utilize all the other rendering functions that have been implemented in order to sample rays,
    pass those rays to the NeRF model to get color and density predictions, and then use volume rendering to create
    an image of this view. 

    Hints: 
    ---
    It is a good idea to "flatten" the height/width dimensions of the data when passing to the NeRF (maintain the color
    channel dimension) and then "unflatten" the outputs. 
    To avoid running into memory limits, it's recommended to use the given get_minibatches() helper function to 
    divide up the input into chunks. For each minibatch, supply them to the model and then concatenate the corresponding
    output vectors from each minibatch to form the complete output vectors. 
    
    Args
    ---
    height (int): 
        the pixel height of an image.
    width (int): the pixel width of an image.
    intrinsics (torch.tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    depth_samples_per_ray (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    encoding_function (Callable): The function used to encode the query points (e.g. positional encoding)
    model (NerfModel): The NeRF model that will be used to render this image
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    rgb_predicted (torch.tensor): 
        A tensor of shape (height, width, num_channels) with the color info at each pixel.
    depth_predicted (torch.tensor): A tensor of shape (height, width) containing the depth from the camera at each pixel.
    """

    rgb_predicted, depth_predicted = None, None
    device = tform_cam2world.device
    ##########################################################################
    # Student code begins here
    ##########################################################################
    cam_ray_origin, cam_ray_dir = get_rays(height, width, intrinsics, tform_cam2world)
    
    # print("cam_ori_shape: ",  cam_ray_origin.shape, " cam_dir_shape: ", cam_ray_dir.shape)
    # cam_ray_origin_flat = cam_ray_origin.reshape(-1, 3)
    # cam_ray_dir_flat = cam_ray_dir.reshape(-1, 3)
    

    view_dir_pts = get_viewing_direction(cam_ray_origin, cam_ray_dir, tform_cam2world)

    
    all_rgb = []
    all_depth = []

    # total_rays = height * width
    rows_per_batch=5

    for i in range(0, height, rows_per_batch):
        batch_end = min(i + rows_per_batch, height)
        batch_height = batch_end - i


        batch_origins = cam_ray_origin[i:batch_end]
        batch_dir = cam_ray_dir[i:batch_end]
        batch_view_dir_pts = view_dir_pts[i:batch_end]
        # actual_batch_size = batch_origins.shape[0]

        # batch_origins_2d = batch_origins.unsqueeze(1)
        # batch_dirs_2d = batch_dir.unsqueeze(1)

        point_samples, depth_values = sample_points_from_rays(batch_origins, batch_dir, 
                                                              near_thresh, far_thresh, 
                                                              depth_samples_per_ray, rand)
        point_samples_flat = point_samples.reshape(-1, 3)
        encoded_points = encoding_function_pts(point_samples_flat)

        # view_dir_pts = get_viewing_direction(batch_origins, batch_dir, tform_cam2world)
        
        view_dir_expand = batch_view_dir_pts.unsqueeze(2).expand(batch_height, width, depth_samples_per_ray, 2)
        view_dir_flat = view_dir_expand.reshape(-1, 2)
        encoded_view_direction = encoding_function_view(view_dir_flat).to(device)
        encoded_input = torch.cat([encoded_points, encoded_view_direction], dim=-1)
        
        batch_outputs_rgb = []
        batch_outputs_sigma = []
        minibatches = get_minibatches(encoded_input, chunksize=1024*8)
        for minibatch in minibatches:
            rgb, sig = model(minibatch)
            batch_outputs_rgb.append(rgb)
            batch_outputs_sigma.append(sig)
        
        color_output = torch.cat(batch_outputs_rgb, dim=0)
        color_output = color_output.reshape(batch_height, width, depth_samples_per_ray, 3)
        # print("color_output_shape: ", color_output.shape)
        sigma = torch.cat(batch_outputs_sigma, dim=0)
        sigma = sigma.reshape(batch_height, width, depth_samples_per_ray)
        # print("sigma_shape: ", sigma.shape)


        weights = compute_compositing_weights(sigma, depth_values)
        # print("weight shape: ", weights.shape, " | ", weights.unsqueeze(-1).shape)
        
        batch_rgb_predicted = torch.sum(weights.unsqueeze(-1)*color_output,dim=-2)
        batch_depth_predicted = torch.sum(weights*depth_values, dim=-1)
        # print(batch_rgb_predicted.shape, " depth shape: ", batch_depth_predicted.shape)

        all_rgb.append(batch_rgb_predicted)
        all_depth.append(batch_depth_predicted)
    
    rgb_predicted = torch.cat(all_rgb, dim=0)
    depth_predicted = torch.cat(all_depth, dim=0)

    
    ##########################################################################
    # Student code ends here
    ##########################################################################

    return rgb_predicted, depth_predicted



def render_rays_batched(rays_origins: torch.Tensor, rays_directions: torch.Tensor, tform_cam2world: torch.Tensor,
                      near_thresh: float, far_thresh: float, depth_samples_per_ray: int,
                      encoding_function_pts: Callable, encoding_function_view: Callable, 
                      model:NerfModel, rand:bool=False, chunk_size:int=1024)-> tuple[torch.Tensor, torch.Tensor]:
    """ This function will utilize all the other rendering functions that have been implemented in order to sample rays,
    pass those rays to the NeRF model to get color and density predictions, and then use volume rendering to create
    an image of this view. 

    Hints: 
    ---
    It is a good idea to "flatten" the height/width dimensions of the data when passing to the NeRF (maintain the color
    channel dimension) and then "unflatten" the outputs. 
    To avoid running into memory limits, it's recommended to use the given get_minibatches() helper function to 
    divide up the input into chunks. For each minibatch, supply them to the model and then concatenate the corresponding
    output vectors from each minibatch to form the complete output vectors. 
    
    Args
    ---
    rays_origins : (N, 3)
    rays_direction : (N, 3)
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    depth_samples_per_ray (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    encoding_function (Callable): The function used to encode the query points (e.g. positional encoding)
    model (NerfModel): The NeRF model that will be used to render this image
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    rgb_predicted (torch.tensor): 
        A tensor of shape (N, num_channels) with the color info at each pixel.
    depth_predicted (torch.tensor): A tensor of shape (N) containing the depth from the camera at each pixel.
    """
    rgb_predicted, depth_predicted = None, None
    device = tform_cam2world.device
    
    ##########################################################################
    # Student code begins here
    ##########################################################################
    
    # print("cam_ori_shape: ",  cam_ray_origin.shape, " cam_dir_shape: ", cam_ray_dir.shape)
    # cam_ray_origin_flat = cam_ray_origin.reshape(-1, 3)
    # cam_ray_dir_flat = cam_ray_dir.reshape(-1, 3)
    

    view_dir_pts = get_viewing_direction(rays_origins, rays_directions, tform_cam2world)

    N = rays_origins.shape[0]
    all_rgb = []
    all_depth = []


    for i in range(0, N, chunk_size):
        batch_end = min(i + chunk_size, N)
        batch_size = batch_end - i

        batch_origins = rays_origins[i:batch_end]
        batch_dir = rays_directions[i:batch_end]
        batch_view_dir_pts = view_dir_pts[i:batch_end]

        # actual_batch_size = batch_origins.shape[0]

        # batch_origins_2d = batch_origins.unsqueeze(1)
        # batch_dirs_2d = batch_dir.unsqueeze(1)

        point_samples, depth_values = sample_points_from_rays_general(batch_origins, batch_dir, 
                                                              near_thresh, far_thresh, 
                                                              depth_samples_per_ray, rand)
        point_samples_flat = point_samples.reshape(-1, 3)
        encoded_points = encoding_function_pts(point_samples_flat)

        # view_dir_pts = get_viewing_direction(batch_origins, batch_dir, tform_cam2world)
        # dirs_flat = batch_view_dir_pts[:, None, :].expand(point_samples.shape).reshape(-1, 3)
        view_dir_expand = batch_view_dir_pts.unsqueeze(1).expand(batch_size, depth_samples_per_ray, 2)
        view_dir_flat = view_dir_expand.reshape(-1, 2)
        encoded_view_direction = encoding_function_view(view_dir_flat).to(device)
        encoded_input = torch.cat([encoded_points, encoded_view_direction], dim=-1)
        
        batch_outputs_rgb = []
        batch_outputs_sigma = []
        minibatches = get_minibatches(encoded_input, chunksize=1024*8)
        for minibatch in minibatches:
            rgb, sig = model(minibatch)
            batch_outputs_rgb.append(rgb)
            batch_outputs_sigma.append(sig)
        
        color_output = torch.cat(batch_outputs_rgb, dim=0)
        color_output = color_output.reshape(batch_origins.shape[0], depth_samples_per_ray, 3) 
        # color_out.shape = (rays_per_batch, depth_samples_per_ray, 3)
        
        sigma = torch.cat(batch_outputs_sigma, dim=0)
        sigma = sigma.reshape(batch_origins.shape[0], depth_samples_per_ray)
       

        weights = compute_compositing_weights(sigma, depth_values)
         
        batch_rgb_predicted = torch.sum(weights.unsqueeze(-1)*color_output,dim=-2)
        batch_depth_predicted = torch.sum(weights*depth_values, dim=-1)
        
        all_rgb.append(batch_rgb_predicted)
        all_depth.append(batch_depth_predicted)
    
    rgb_predicted = torch.cat(all_rgb, dim=0)
    depth_predicted = torch.cat(all_depth, dim=0)

    # raise NotImplementedError('`render_image_nerf()` function needs to be implemented')
    
    ##########################################################################
    # Student code ends here
    ##########################################################################

    return rgb_predicted, depth_predicted



def render_full_image_chunked(height: int, width: int, intrinsics: torch.Tensor, tform_cam2world: torch.Tensor,
                              near_thresh: float, far_thresh: float, depth_samples_per_ray: int,
                              encoding_function_pts: Callable, encoding_function_view: Callable, 
                              model:NerfModel, rand:bool=False, chunk_size:int=1024)-> tuple[torch.Tensor, torch.Tensor]:
    
    with torch.no_grad():
        rays_o, rays_d = get_rays(height, width, intrinsics, tform_cam2world)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        rgb_predicted, depth_predicted = render_rays_batched(rays_o, rays_d, tform_cam2world, 
                                                    near_thresh, far_thresh, depth_samples_per_ray, 
                                                    encoding_function_pts, encoding_function_view , model, rand, chunk_size)

        rgb_predicted = rgb_predicted.reshape(height, width, 3)
        depth_predicted = depth_predicted.reshape(height, width)
    
    return rgb_predicted, depth_predicted