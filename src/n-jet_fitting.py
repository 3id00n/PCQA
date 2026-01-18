import torch, sys, os
import numpy as np
sys.path.insert(0, './utils')
#sys.path.insert(0, './models')
from models import DeepFit
import tutorial_utils as tu
from tqdm import tqdm

jet_order_fit = 3
device = torch.device("cuda")   # force CPU

def run():
    fileList = ['./src/data/bunny.xyz']
    
    for filename in tqdm(fileList):
        print(f'# {filename}')
        filename = filename[:filename.find('.xyz')]
        point_cloud_dataset = tu.SinglePointCloudDataset(f'{filename}.xyz', points_per_patch=256)
        dataloader = torch.utils.data.DataLoader(point_cloud_dataset, batch_size=2048, num_workers=4, shuffle=False, pin_memory=False)
        
        for batchind, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            points, data_trans, scale_radius = data
            points = points.to(device)
            data_trans = data_trans.to(device)
            scale_radius = scale_radius.to(device)

            beta, n_est, neighbors_n_est = DeepFit.fit_Wjet(
                points,
                torch.ones_like(points[:, 0]),
                order=jet_order_fit,
                compute_neighbor_normals=False
            )
            n_est = n_est.detach().cpu()
            beta = beta.detach().cpu()
            normals = n_est if batchind == 0 else torch.cat([normals, n_est], 0)

            if beta.dim() == 1 and jet_order_fit == 3:
                beta = beta.reshape(-1, 10)
            elif beta.dim() == 1 and jet_order_fit == 4:
                beta = beta.reshape(-1, 15)
            betas = beta if batchind == 0 else torch.cat([betas, beta], 0)

        newPointWithNormal = np.concatenate(
            [point_cloud_dataset.rawPoints[:, 0:3], normals, betas], axis=1
        )
        print(newPointWithNormal.shape)
        saveFilename = f'{filename}_order{jet_order_fit}_normal_beta.txt'
        np.savetxt(saveFilename, newPointWithNormal, fmt='%1.6f')
        print(f'save to {saveFilename}')

if __name__ == '__main__':
    run()
