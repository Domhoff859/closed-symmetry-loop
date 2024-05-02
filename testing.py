import torch
from network.cnn_definition_paper import rgb255_to_obj_net 
from torch.nn import Lambda
from xnp.xnp_torch_layers import MtMCreation, MtMEvaluation
from reverse_op import PODestarisation
from data_handling.model_info import load_model_info
from data_handling import DataLoader

import torch.nn as nn
import torchvision.models as models

# Set the paths and parameters
bop_path  = '/export/jesse/BOP'
dataset = 'tless'
dataset_path = f'{bop_path}/{dataset}'

results_path = '../results/'

foreign_info = 'scene_maskrcnn_detections_syn+real_info.json'

test = ['test_primesense']

xyDim = 112
strides = 2

# Define a helper function
def totuplething(x1, x2, x4, x7):
    return ((x1, x2, x4, x7), x7)

# Loop over a range of values
for oiu in range(1,31):
    # Load the model info
    model_info = load_model_info(dataset_path, oiu, verbose=1)

    # Load the trained CNN
    trained_cnn = torch.load(f'{dataset_path}/saved_models/csl_o{oiu}_trainable_layers')

    # Extract inputs and outputs from the trained CNN
    rgb_input, depth_input, cam_K, coord_K = trained_cnn.inputs
    star_representation, dash_representation, w_px, w_d, seg = trained_cnn.outputs

    # Perform some operations on the inputs and outputs
    segmentation = Lambda(lambda x: torch.cast(x>0.75, x.dtype))(seg)
    w_px = Lambda(lambda x: x[0][...,None,:] * x[1][...,None])([w_px, segmentation])
    w_d = Lambda(lambda x: x[0][...,None,:] * x[1][...,None])([w_d, segmentation])

    # Perform PODestarisation
    po_image = PODestarisation(model_info,amount_of_instances = 1)(star_representation, dash_representation, segmentation)
    po_image = Lambda(lambda x: x[0][...,None,:] * x[1][...,None])([po_image, segmentation])
    MtM = MtMCreation("po", strides)((po_image, w_px, cam_K, coord_K))
    pred_T_, pred_Omega = MtMEvaluation()(MtM)

    # Perform some more operations
    depth = Lambda(lambda x: x[:,::strides,::strides])(depth_input)
    MtM_d = MtMCreation("depth")((po_image, depth , w_d))
    pred_T_d, pred_Omega = MtMEvaluation()(torch.add(MtM, MtM_d))

    # Create a prediction model
    pred_model = nn.Sequential(trained_cnn, nn.Identity())

    # Load foreign data for testing
    test_foreign_data = DataLoader.load_foreign_data([f'{dataset_path}/{d}' for d in test ], foreign_info, oiu)
    print(f'Found TEST foreign data for {len(test_foreign_data)} occurencies of object {oiu}, where {len([d for d in test_foreign_data if "primesense" in d["root"]])} origined from primesense.')

    # Make predictions using the model
    res = pred_model(DataLoader.Dataset(test_foreign_data,xyDim, test_mode=True).batch(1).prefetch(20).map(totuplething))

    # Prepare the results for writing to a file
    rowwisestr = 'scene_id,im_id,obj_id,score,R,t,time\n'
    rowwisestr_d = 'scene_id,im_id,obj_id,score,R,t,time\n'
    for tdi, td in enumerate(test_foreign_data):
        pose_ = res[0][tdi][0]
        pose_d = res[1][tdi][0]
        rowwisestr += f'{int(td["root"][-6:])},{int(td["file_name"])},{oiu},0.5,{" ".join(list(pose_[:3,:3].flatten().astype(str)))},{" ".join(list(pose_[:3,3].flatten().astype(str)))},-1\n'
        rowwisestr_d += f'{int(td["root"][-6:])},{int(td["file_name"])},{oiu},0.5,{" ".join(list(pose_d[:3,:3].flatten().astype(str)))},{" ".join(list(pose_d[:3,3].flatten().astype(str)))},-1\n'
    
    # Write the results to a file
    f = open(f"{results_path}/new-paper-{oiu}-cosypose-frame-rgb_tless-test.csv","w")
    f.write(rowwisestr)
    f.close()

    f = open(f"{results_path}/new-paper-{oiu}-cosypose-frame-rgb-d_tless-test.csv","w")
    f.write(rowwisestr_d)
    f.close()
    
# Combine the results from multiple files
combined_results = 'scene_id,im_id,obj_id,score,R,t,time\n'
for oiu in range(1,31):
    f = open(f"{results_path}/new-paper-{oiu}-cosypose-frame-rgb_tless-test.csv","r")
    combined_results += f.read()[len('scene_id,im_id,obj_id,score,R,t,time\n'):]
    f.close()
    
# Write the combined results to a file
f = open(f"{results_path}/new-paper-1-30-cosypose-frame-rgb_tless-test.csv","w")
f.write(combined_results)
f.close()

combined_results = 'scene_id,im_id,obj_id,score,R,t,time\n'
for oiu in range(1,31):
    f = open(f"{results_path}/new-paper-{oiu}-cosypose-frame-rgb-d_tless-test.csv","r")
    combined_results += f.read()[len('scene_id,im_id,obj_id,score,R,t,time\n'):]
    f.close()
    
# Write the combined results to a file
f = open(f"{results_path}/new-paper-1-30-cosypose-frame-rgb-d_tless-test.csv","w")
f.write(combined_results)
f.close()
