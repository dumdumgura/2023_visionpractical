# 2023_vision_practical
1. Data Preprocessing: set mode to 'siren_sdf' or 'sdf' to generate the data.
   mode 'siren_sdf' is used for sdf representation, together with loss_formulation in siren paper.
   mode 'sdf' is used for sdf or sdf representation, together with L1_loss or BCE_loss.
   
   
```python
   python utils/mesh_to_sdf_test.py --mode=siren_sdf --input_folder=$YOUR_INPUT_FOLDER --output_folder=$YOUR_OUTPUT_FOLDER

```
   
