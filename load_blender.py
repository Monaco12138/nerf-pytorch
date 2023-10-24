import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


# 这个表示沿z轴位移的矩阵
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

# 这个表示沿x轴旋转 phi 弧度
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


# half_res: 以一半的分辨率加载图像
def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            # shape: h, w, c(RGBA)
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append( counts[-1] + imgs.shape[0] )
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    #counts: [0, 100, 113, 138]
    # 0~99: train
    # 100~112: val
    # 113~137: test

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    #imgs: [138, h, w, 4]
    #poses: [138, 4, 4]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    print( imgs.shape )
    print( poses.shape )

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    #print( camera_angle_x )

    #focal = 1111.111
    # camera_angle_x 表示水平FOV
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # imgs: [138, h, w, 4], poses:[138, 4, 4], render_poses:[40,4,4]
    return imgs, poses, render_poses, [H, W, focal], i_split


if __name__ == '__main__':
    load_blender_data('./data/nerf_synthetic/lego', True, 8)

    # img = cv2.imread( '/home/ubuntu/data/home/main/nerf-pytorch/data/nerf_synthetic/lego/train/r_4.png', cv2.IMREAD_UNCHANGED )
    # print(img[400:440,400:440,-1])
    # #cv2.imwrite( './test.png', img )