# Adaptive Grids for Neural Scene Representation

Official implementation of **Adaptive Grids for Neural Scene Representation** presented at **VMV 2024**.  
**[Project Page](https://vcai.mpi-inf.mpg.de/projects/agrids/)** | **[Paper](https://pure.mpg.de/rest/items/item_3624536/component/file_3624537/content)**

<img src="assets/adaptive-grid.png" alt="Adaptive Grid" width="700">


![Adaptive Grid Video](assets/dvgo_comparison.gif)  

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

Follow these steps to train the model:

1. **Prepare the Dataset**

Download the required datasets and place them in the `data` folder. Below are the links to the datasets:

- **NeRF Synthetic**: [Download here](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)
- **NeRF LLFF Data**: [Download here](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)
- **Synthetic NSVF**: [Download here](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)

2. **Training**: Use the following command to start training:

    ```bash
    python train.py --config configs/your_config.yaml
    ```

    Make sure to replace `your_config.yaml` with the appropriate configuration file from the `configs` folder.

### Inference

For inference, use the `--render_only` flag and run the same command as training.

```bash
python train.py --config configs/your_config.yaml --render_only
```


## Acknowledgements

This codebase is based on the [DirectVoxGO (DVGO)](https://github.com/sunset1995/DirectVoxGO) implementation by Sunset1995. 



## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{pajoum2024adaptive,
  title={Adaptive Grids for Neural Scene Representation},
  author={Pajoum, Barbod and Fox, Gereon and Elgharib, Mohamed and Habermann, Marc and Theobalt, Christian},
  booktitle={International Symposium on Vision, Modeling, and Visualization},
  year={2024},
  organization={Eurographics Association}
}
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

