import torch
from tqdm import tqdm 
import numpy as np
import os
from pytorch3d.structures import Pointclouds
from typing import cast
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

def xyz_multiple_cameras(depths, cameras):

    device = depths.device
    depth = depths.clone()

    pixel_center = 0.5 

    H, W = depth.shape[1], depth.shape[2]

    fx, fy, cx, cy = (cameras.focal_length[:,0], cameras.focal_length[:,1], 
                        cameras.principal_point[0,0], cameras.principal_point[0,1])

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device) + pixel_center,
        torch.arange(H, dtype=torch.float32, device=device) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack(
        [-(i - cx) * depth / fx.view(depth.shape[0], 1, 1), -(j - cy) * depth / fy.view(depth.shape[0], 1, 1), depth], -1
    )

    # Transform the generated points from view to the world coordinates
    xy_depth_world = cameras.get_world_to_view_transform().inverse().transform_points(directions.float().reshape(depth.shape[0], H*W ,3))
    return xy_depth_world

def compute_scale_shift(duster_depths, da_depths, steps=1000, lr=0.01, device='cpu'):
    scale = torch.nn.Parameter(
        torch.ones(duster_depths.shape[0], device=device, dtype=torch.float)
    )
    shift = torch.nn.Parameter(
        torch.zeros(da_depths.shape[0], device=device, dtype=torch.float)
    )

    mask = (duster_depths > 0) * (da_depths > 0) 

    duster_depth_masked = (duster_depths * mask).to(device)
    da_depth_masked = (da_depths * mask).to(device)

    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([scale, shift], lr=lr)

    for i in tqdm(range(steps), desc="Computing scale and shift"):
        optimizer.zero_grad()
        loss = mse_loss(
            scale[...,None,None] * da_depth_masked + shift[...,None,None], duster_depth_masked
        )
        loss.backward()
        optimizer.step()
        
    return scale, shift


import trimesh
def get_pc(imgs, pts3d, mask):
    imgs = (imgs).cpu().numpy()
    pts3d = (pts3d).cpu().numpy()
    mask = (mask).view(pts3d.shape[0],-1).cpu().numpy()
    
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    
    pts = pts.reshape(-1, 3)[::3]
    col = col.reshape(-1, 3)[::3]
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts


def save_pointcloud_with_normals(imgs, pts3d, msk, sparse_path, name = ''):
    os.makedirs(sparse_path, exist_ok=True)
    pc = get_pc(imgs, pts3d, msk)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    save_path = sparse_path / f'points3D{name}.ply'

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))
            
def merge_pointclouds(point_clouds):    
    points = torch.cat([pc.points_padded() for pc in point_clouds], dim=1)
    features = torch.cat([pc.features_padded() for pc in point_clouds], dim=1)
    return Pointclouds(points=[points[0]], features=[features[0]])

def render_image_new_cam(camera, pc, radius=0.01, weights_precision='default'):

    raster_settings = PointsRasterizationSettings(
            image_size=(int((camera.image_size[:,0][0])), int((camera.image_size[:,1][0]))),
            radius = radius,
            #radius = (1 / float(max(cameras[0].image_size[0,0], cameras[0].image_size[0,1])) * 2.0) if radius is None else radius,
            points_per_pixel = 16,
            bin_size=0,
    )

    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)

    renderer = PointsRendererWithMasks( 
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    render_img, render_mask, render_depth, _ = renderer(pc, weights_precision=weights_precision)

    return render_img, render_mask, render_depth

class PointsRendererWithMasks(PointsRenderer):
    def forward(self, point_clouds, weights_precision = 'default' , **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists
        #weights = torch.ones_like(dists2)#-> commentata questa riga e aggiunta la riga successiva, presente nel codice originale di pytorch3d
        #l'implemntazione di invisible stitch eliminava il gradiente dall'equazione togliendo dist2
        if weights_precision == 'default':
            weights = 1 - dists2 / (r * r)
        elif weights_precision == 'coarse':
            weights = torch.ones_like(dists2)

        ok = cast(torch.BoolTensor, (fragments.idx >= 0)).float()

        weights = weights * ok 

        fragments_prm = fragments.idx.long().permute(0, 3, 1, 2)
        weights_prm = weights.permute(0, 3, 1, 2)
        images = self.compositor(
            fragments_prm,
            weights_prm,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        cumprod = torch.cumprod(1 - weights, dim=-1)
        cumprod = torch.cat((torch.ones_like(cumprod[..., :1]), cumprod[..., :-1]), dim=-1)
        depths = (weights * cumprod * fragments.zbuf).sum(dim=-1)

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)
        #masks = fragments.idx.long()[..., 0] >= 0 * weights.sum(dim=-1)
        masks = (fragments.idx.long()[..., 0] >= 0).float() #* (weights.sum(dim=-1) > weights.sum(dim=-1).mean()).float()
        #masks = -torch.prod(1.0 - weights, dim=-1) + 1.0
        
        weights_sum = weights.sum(dim=-1)

        # Normalize weights_sum to the range [0, 1]
        normalized_weights = (weights_sum - weights_sum.min()) / (weights_sum.max() - weights_sum.min())

        return images, masks, depths, normalized_weights
    

def render_image_with_slerp(cameras, interpolation_factor, radius, cameras_idx, ppp = 16, background_color = (0,0,0), weights_precision = 'default',  pc=None):
    
    q0 = matrix_to_quaternion(cameras[cameras_idx[:-1]].R)
    q1 = matrix_to_quaternion(cameras[cameras_idx[1:]].R)

    t = interpolation_factor  # Interpolation factor

    slerp_result = slerp(q0, q1, t)

    # Assuming cameras[0].T and cameras[1].T are the translations to interpolate
    T0 = cameras[cameras_idx[:-1]].T.squeeze(0)
    T1 = cameras[cameras_idx[1:]].T.squeeze(0)

    # Linear interpolation for translation
    interpolated_T = (1 - t) * T0 + t * T1
    H, W = cameras[0].image_size[0,0].int().item(), cameras[0].image_size[0,1].int().item()
    
    interpolated_camera = PerspectiveCameras(R=quaternion_to_matrix(slerp_result), T=interpolated_T.unsqueeze(0),
                                            focal_length=cameras.focal_length[0].unsqueeze(0) , principal_point=cameras[0].principal_point, 
                                            image_size=cameras[0].image_size.int(), device=cameras.device, in_ndc=False)

    raster_settings = PointsRasterizationSettings(
                image_size=(H, W),
                radius = radius,
                points_per_pixel = ppp,
                bin_size = 0,
                #radius = 1e-2,
                #points_per_pixel = 1
    )

    rasterizer = PointsRasterizer(cameras=interpolated_camera, raster_settings=raster_settings)

    renderer = PointsRendererWithMasks( 
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=background_color)
    )

    render_img, render_mask, render_depth, weights = renderer(pc, weights_precision=weights_precision)
    #img = Image.fromarray((render_img.detach().cpu().numpy() * 255).astype(np.uint8))
    return render_img, render_mask, render_depth, interpolated_camera,weights


def slerp(q0, q1, t):
    # Ensure inputs are all normalized quaternions
    q0 = q0 / torch.norm(q0, dim=1)
    q1 = q1 / torch.norm(q1, dim=1)
    
    # Compute the cosine of the angle between the quaternions
    dot = (q0 * q1).sum(dim=1)
    
    # If the dot product is negative, slerp won't take the shorter path.
    # So we'll adjust the signs to ensure it does.
    dot_mask = dot<0
    q1[dot_mask,:] = -q1[dot_mask,:]
    dot[dot_mask] = -dot[dot_mask]

    # Clamp dot product to stay within domain of acos()
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Calculate coefficients
    theta_0 = torch.acos(dot)  # theta_0 = angle between input vectors
    sin_theta_0 = torch.sin(theta_0)  # compute this value only once
    
    sin_mask = (sin_theta_0 > 1e-6)
    
    s0 = torch.zeros_like(theta_0)
    s1 = torch.zeros_like(theta_0)

    theta = (theta_0[sin_mask] * t) # theta = angle between v0 and result
    sin_theta = torch.sin(theta)  # compute this value only once
    
    s0[sin_mask] = torch.cos(theta) - dot[sin_mask] * sin_theta / sin_theta_0

    s1[sin_mask] = sin_theta / sin_theta_0[sin_mask]
    
    s0[~sin_mask] = 1.0 - t
    s1[~sin_mask] = t

    return (s0.unsqueeze(1) * q0) + (s1.unsqueeze(1) * q1)