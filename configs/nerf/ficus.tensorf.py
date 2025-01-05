_base_ = '../default.py'

expname = 'dvgo_ficus_tensorf'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/ficus',
    dataset_type='blender',
    white_bkgd=True,
)

fine_train = dict(
    lrate_density=0.02,
    lrate_k0=0.02,
    pg_scale=[1000, 2000, 3000, 4000, 5000, 6000],
)


fine_model_and_render = dict(
        num_voxels=200**3,
    density_type='TensoRFGrid',
    density_config=dict(n_comp=3),
    k0_type='TensoRFGrid',
    k0_config=dict(n_comp=6),
)

