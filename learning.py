import tensorflow as tf

from data_handling.model_info import load_model_info
from data_handling import DataLoader, Conversion_Layers
from network import loss, cnn_definition_paper

from star_representation import StarRepresentation
from dash_repesentation import RemoveCameraEffect, DashRepresentation
from reverse_op import PODestarisation

from keras.optimizers import Adam

dataset = 'tless'

dataset_path = f'{dataset}'

train = ['train_primesense', 'train_pbr']
test = ['test_primesense']

xyDim = 112
strides = 2


def pred_loss(true, pred):
    return pred

def totuplething(x1, x2, x3, x4, x5, x6, x7):
    return ((x1, x2, x3, x4, x7, x5, x6), x6)

for oiu in range(1,30):
    print('Object ', oiu)
    model_info = load_model_info(dataset_path, oiu, verbose=1)
    train_data = DataLoader.load_gt_data([f'{dataset_path}/{d}' for d in train ], oiu)
    print(f'Found train data for {len(train_data)} occurencies of object {oiu}, where {len([d for d in train_data if "primesense" in d["root"]])} origined from primesense.')

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():    
        inputs, valid_po, isvalid, depth, segmentation = Conversion_Layers.create_Dataset_conversion_layers(xyDim, xyDim, model_info, strides)

        valid_dash = DashRepresentation(model_info["symmetries_discrete"][0][:3,-1] / 2. if len(model_info["symmetries_discrete"]) > 0 else 0 )(inputs['roationmatrix'], valid_po)
        valid_po_star = StarRepresentation(model_info)(valid_po)

        cnn_po_star, cnn_po_dash, cnn_w_px, cnn_w_d, cnn_seg = cnn_definition_paper.rgb255_to_obj_net(inputs['rgb'])
        dash_image = RemoveCameraEffect(strides)(cnn_po_dash, inputs['camera_matrix'], inputs['coord_offset'])
        
        csl_trainable_layers = tf.keras.Model([inputs['rgb'], inputs['depth'], inputs['camera_matrix'], inputs['coord_offset']],
                                              [cnn_po_star, dash_image, cnn_w_px, cnn_w_d, cnn_seg])
                
        po_image = PODestarisation(model_info,amount_of_instances = 1)(cnn_po_star, dash_image, isvalid, inputs['roationmatrix'])
        po_uv, po_cam = loss.Po_to_Img()(po_image, inputs['camera_matrix'], inputs['roationmatrix'], inputs['translation'])

    #     diff_po = Lambda(squared_diff_of_pos, name='po_diff')((po_image, valid_po, isvalid))
        diff_postar = loss.AvgSqrDiff_of_validPixels(name='pos_diff')(cnn_po_star, valid_po_star, isvalid)
        diff_vo = loss.AvgSqrDiff_of_validPixels(name='vo_diff')(dash_image, valid_dash, isvalid)
        (seg_loss, seg_met, seg_fgmet) = loss.Seg_Loss(name='seg')(cnn_seg, segmentation)  

        sig2inv = loss.ToOmega()(cnn_w_px, isvalid)
        po_uv_diff = loss.UV_diff(strides)(po_uv, inputs['coord_offset'])
        lw2_loss, chi2error = loss.Avg_nllh(name='w2')(sig2inv, po_uv_diff, isvalid)

        sig1inv =  loss.ToOmega()(cnn_w_d, isvalid)
        po_depth_diff = loss.D_diff()(po_cam, depth)
        lw1_loss, chi2error_d = loss.Avg_nllh(name='w1')(sig1inv, po_depth_diff, isvalid)
#         lw1_loss, chi2error_d = Lambda(wp_loss_wd, name='w1')((cnn_w_d, po_cam, depth, isvalid))

        train_povoseg_model = tf.keras.Model(inputs.values(), (diff_postar, diff_vo, seg_loss, seg_met, seg_fgmet))
        train_model = tf.keras.Model(inputs.values(), (diff_postar, diff_vo, seg_loss, seg_met, seg_fgmet,
                                                      lw2_loss, chi2error, lw1_loss, chi2error_d))

        train_povoseg_model.compile(Adam(0.0001,  amsgrad=True),
                            loss = pred_loss,
                            loss_weights=(1,1,1,0,0)
                          )

        train_model.compile(Adam(0.0001,  amsgrad=True),
                            loss = pred_loss,
                            loss_weights=(1,1,1,0,0,
                                          1,0,1,0)
                           ) 
        
        train_povoseg_model.fit(DataLoader.Dataset(train_data,xyDim, times=2, group_size=5, random=True).batch(80).prefetch(20).map(totuplething),
                        epochs=2,
                        verbose=1,
                        workers=8,
                        max_queue_size=100,
                        use_multiprocessing=True)
        train_povoseg_model.save_weights(f'{dataset_path}/saved_weights/new_{oiu}_train_povoseg_2e')

        train_model.fit(DataLoader.Dataset(train_data,xyDim, times=2, group_size=5, random=True).batch(40).prefetch(20).map(totuplething),
                                epochs=10,
                                verbose=1,
                                workers=8,
                                max_queue_size=100,
                                use_multiprocessing=True)
        train_model.save_weights(f'{dataset_path}/saved_weights/csl_o{oiu}_train_model_10e')
        csl_trainable_layers.save(f'{dataset_path}/saved_models/csl_o{oiu}_trainable_layers')