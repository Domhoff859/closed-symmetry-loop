import torch
from data_handling.model_info import load_model_info
from data_handling import DataLoader, Conversion_Layers
from network import loss, cnn_definition_paper
from star_representation import StarRepresentation
from dash_repesentation import RemoveCameraEffect, DashRepresentation
from reverse_op import PODestarisation

import torch.nn as nn
import torch.optim as optim

# Define the dataset and its path
dataset = 'tless'
dataset_path = f'{dataset}'

# Define the train and test data
train = ['train_primesense', 'train_pbr']
test = ['test_primesense']

# Define the dimensions and strides
xyDim = 112
strides = 2

# Define a custom loss function
def pred_loss(true, pred):
    return pred

# Define a helper function
def totuplething(x1, x2, x3, x4, x5, x6, x7):
    return ((x1, x2, x3, x4, x7, x5, x6), x6)

# Loop over a range of values
for oiu in range(1,30):
    print('Object ', oiu)

    # Load model information
    model_info = load_model_info(dataset_path, oiu, verbose=1)

    # Load train data
    train_data = DataLoader.load_gt_data([f'{dataset_path}/{d}' for d in train ], oiu)
    print(f'Found train data for {len(train_data)} occurencies of object {oiu}, where {len([d for d in train_data if "primesense" in d["root"]])} origined from primesense.')

    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the inputs and valid_po tensors
    inputs, valid_po, isvalid, depth, segmentation = Conversion_Layers.create_Dataset_conversion_layers(xyDim, xyDim, model_info, strides)

    # Create the valid_dash and valid_po_star tensors
    valid_dash = DashRepresentation(model_info["symmetries_discrete"][0][:3,-1] / 2. if len(model_info["symmetries_discrete"]) > 0 else 0 )(inputs['roationmatrix'], valid_po)
    valid_po_star = StarRepresentation(model_info)(valid_po)

    # Define the CNN layers
    cnn_po_star, cnn_po_dash, cnn_w_px, cnn_w_d, cnn_seg = cnn_definition_paper.rgb255_to_obj_net(inputs['rgb'])
    dash_image = RemoveCameraEffect(strides)(cnn_po_dash, inputs['camera_matrix'], inputs['coord_offset'])

    # Create the csl_trainable_layers module
    csl_trainable_layers = nn.Module([inputs['rgb'], inputs['depth'], inputs['camera_matrix'], inputs['coord_offset']],
                                      [cnn_po_star, dash_image, cnn_w_px, cnn_w_d, cnn_seg])

    # Perform PODestarisation
    po_image = PODestarisation(model_info,amount_of_instances = 1)(cnn_po_star, dash_image, isvalid, inputs['roationmatrix'])
    po_uv, po_cam = loss.Po_to_Img()(po_image, inputs['camera_matrix'], inputs['roationmatrix'], inputs['translation'])

    # Calculate the differences and losses
    diff_postar = loss.AvgSqrDiff_of_validPixels(name='pos_diff')(cnn_po_star, valid_po_star, isvalid)
    diff_vo = loss.AvgSqrDiff_of_validPixels(name='vo_diff')(dash_image, valid_dash, isvalid)
    (seg_loss, seg_met, seg_fgmet) = loss.Seg_Loss(name='seg')(cnn_seg, segmentation)  

    sig2inv = loss.ToOmega()(cnn_w_px, isvalid)
    po_uv_diff = loss.UV_diff(strides)(po_uv, inputs['coord_offset'])
    lw2_loss, chi2error = loss.Avg_nllh(name='w2')(sig2inv, po_uv_diff, isvalid)

    sig1inv =  loss.ToOmega()(cnn_w_d, isvalid)
    po_depth_diff = loss.D_diff()(po_cam, depth)
    lw1_loss, chi2error_d = loss.Avg_nllh(name='w1')(sig1inv, po_depth_diff, isvalid)

    # Create the train_povoseg_model and train_model modules
    train_povoseg_model = nn.ModuleList([inputs.values(), (diff_postar, diff_vo, seg_loss, seg_met, seg_fgmet)])
    train_model = nn.ModuleList([inputs.values(), (diff_postar, diff_vo, seg_loss, seg_met, seg_fgmet,
                                                  lw2_loss, chi2error, lw1_loss, chi2error_d)])

    # Move the modules to the device
    train_povoseg_model.to(device)
    train_model.to(device)

    # Define the optimizers
    optimizer_povoseg = optim.Adam(train_povoseg_model.parameters(), lr=0.0001, amsgrad=True)
    optimizer_model = optim.Adam(train_model.parameters(), lr=0.0001, amsgrad=True)

    # Define the criterion (loss function)
    criterion = nn.MSELoss()

    # Training loop for train_povoseg_model
    for epoch in range(2):
        train_povoseg_model.train()
        for data in DataLoader.Dataset(train_data, xyDim, times=2, group_size=5, random=True).batch(80).prefetch(20).map(totuplething):
            inputs, targets = data
            inputs = [input.to(device) for input in inputs]
            targets = [target.to(device) for target in targets]

            optimizer_povoseg.zero_grad()
            outputs = train_povoseg_model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_povoseg.step()

    # Save the trained model
    torch.save(train_povoseg_model.state_dict(), f'{dataset_path}/saved_weights/new_{oiu}_train_povoseg_2e')

    # Training loop for train_model
    for epoch in range(10):
        train_model.train()
        for data in DataLoader.Dataset(train_data, xyDim, times=2, group_size=5, random=True).batch(40).prefetch(20).map(totuplething):
            inputs, targets = data
            inputs = [input.to(device) for input in inputs]
            targets = [target.to(device) for target in targets]

            optimizer_model.zero_grad()
            outputs = train_model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_model.step()

    # Save the trained models
    torch.save(train_model.state_dict(), f'{dataset_path}/saved_weights/csl_o{oiu}_train_model_10e')
    torch.save(csl_trainable_layers.state_dict(), f'{dataset_path}/saved_models/csl_o{oiu}_trainable_layers')