import torch
import numpy as np
from math import isclose
from utils import *

def get_obj_star0_from_obj_star(obj_star: torch.Tensor,
                                x_factor: int = 1, y_factor: int = 1, z_factor: int = 1) -> torch.Tensor:
    """
    Get the object star0 from the object star by applying scaling factors along the x, y, and z axes.

    Args:
        obj_star (torch.Tensor): The object star tensor.
        x_factor (int): Scaling factor along the x-axis. Default is 1.
        y_factor (int): Scaling factor along the y-axis. Default is 1.
        z_factor (int): Scaling factor along the z-axis. Default is 1.

    Returns:
        torch.Tensor: The object star0 tensor.
    """
    R = torch.eye(3, batch_shape=(obj_star.size())[:-1])
    
    obj_star = change_Angle_around_Axis(R[...,2], obj_star, R[...,0], 1. / z_factor)
    obj_star = change_Angle_around_Axis(R[...,1], obj_star, R[...,2], 1. / y_factor)
    obj_star = change_Angle_around_Axis(R[...,0], obj_star, R[...,1], 1. / x_factor)
    
    return obj_star


class PODestarisation(torch.nn.Module):
    def __init__(self, 
                 model_info: dict,
                 amount_of_instances: int = 6,
                 **kwargs):
        """
        Initialize the PODestarisation module.

        Args:
            model_info (dict): Information about the model.
            amount_of_instances (int): The amount of instances. Default is 6.
            **kwargs: Additional keyword arguments.
        """
        super(PODestarisation, self).__init__(**kwargs)
        self.supports_masking = True
        
        self.model_info = model_info
        self.amount_of_instances = amount_of_instances
        
    def get_config(self) -> dict:
        """
        Get the configuration of the module.

        Returns:
            dict: The configuration dictionary.
        """
        config = {
            'amount_of_instances': self.amount_of_instances,
            'model_info': self.model_info
        }
        base_config = super(PODestarisation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, postar: torch.Tensor, vo: torch.Tensor, iseg: torch.Tensor, train_R: torch.Tensor = None) -> torch.Tensor:
        """
        Perform the call operation of the module.

        Args:
            postar (torch.Tensor): The postar tensor.
            vo (torch.Tensor): The vo tensor.
            iseg (torch.Tensor): The iseg tensor.
            train_R (torch.Tensor): The train_R tensor. Default is None.

        Returns:
            torch.Tensor: The resulting tensor.
        """
        def angle_substraction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """
            Compute the angle substraction between two tensors.

            Args:
                a (torch.Tensor): The first tensor.
                b (torch.Tensor): The second tensor.

            Returns:
                torch.Tensor: The resulting tensor.
            """
            return torch.minimum(torch.abs(a-b), torch.abs(torch.minimum(a,b) + np.pi * 2. - torch.maximum(a,b)))

        def best_symmetrical_po(star0: torch.Tensor, factor: int, axis: torch.Tensor) -> torch.Tensor:
            """
            Find the best symmetrical po tensor.

            Args:
                star0 (torch.Tensor): The star0 tensor.
                factor (int): The scaling factor.
                axis (torch.Tensor): The axis tensor.

            Returns:
                torch.Tensor: The best symmetrical po tensor.
            """
            if train_R is None:
                ref = self.generate_ref_data([postar, vo, iseg], 15,  3)
            else:
                gt_ref_data = {
                    'po': torch.eye(3, batch_shape=((train_R.size())[:1]))[:,None],
                    'vo': torch.transpose(train_R[:,None], perm=[0,1,3,2])
                }
                ref = gt_ref_data
                
            dash_angles = angle_between(vo, ref['vo'], 'byxi, boji->byxoj')


            symR = make_R_from_angle_axis(2*np.pi / factor, axis)

            allp_pos = star0[:,None,:]
            for _ in range(factor-1):
                newp = torch.einsum('ij, byxj -> byxi', symR, allp_pos[:,-1,:])
                allp_pos = torch.concat([allp_pos, newp[:,None,:]], axis=-2)


            allp_po_angles = angle_between(torch.no_grad(allp_pos), ref['po'], 'byxsi, boji->byxosj')
            allp_angle_diffs = torch.sum(angle_substraction(allp_po_angles, dash_angles.unsqueeze(-2))**2, axis=-1)

            arg_min = torch.argmin(allp_angle_diffs, axis=-1)
            best_po = torch.gather(allp_pos, dim=-2, index=arg_min.unsqueeze(-1), batch_dims=3)

            o_wide_error = torch.sum(torch.min(allp_angle_diffs, dim=-1)[0][..., None, :] * iseg[..., None], dim=[1, 2], keepdim=True)
            arg_min = torch.argmin(o_wide_error, dim=-1)
            arg_min_ = arg_min * iseg
            best_po = torch.gather(best_po, dim=-2, index=arg_min_.unsqueeze(-1), batch_dims=3)

            return torch.sum(best_po * iseg[..., None], dim=-2)

        def best_continues_po(star0: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
            """
            Find the best continuous po tensor.

            Args:
                star0 (torch.Tensor): The star0 tensor.
                axis (torch.Tensor): The axis tensor.

            Returns:
                torch.Tensor: The best continuous po tensor.
            """
            if train_R is None:                
                def direct_calc_z_dir(x):
                    postar, vo_image, seg = x
                    def make_equation_matrix_pd(vo, postar):
                        A = vo[...,None,:]
                        AtA = torch.matmul(A,A, transpose_a=True)
                        b = postar[...,2:,None]
                        Atb = torch.matmul(A,b, transpose_a=True)
                        return AtA, Atb

                    AtA, Atb = make_equation_matrix_pd(vo_image * seg, postar*seg)

                    _AtA = torch.sum(AtA,axis=[1,2])
                    _Atb = torch.sum(Atb,axis=[1,2])

                    return torch.matmul(torch.linalg.pinv(_AtA), _Atb)


                def make_oracle_R(Rz):
                    z = normalize(Rz[...,0])
                    o = torch.concat([z[...,1:], z[...,:1] ],axis=-1)
                    x = cross(z,o)
                    y = cross(z,x)
                    z = cross(x,y)
                    return torch.stack([x,y,z], axis=-1)
                
                _R = make_oracle_R(direct_calc_z_dir([postar, vo, iseg]))
                
            else:
                _R = train_R
                
            ref = {
                'po': torch.eye(3, batch_shape=((_R.size())[:1]))[:,None],
                'vo': torch.transpose(_R[:,None], perm=[0,1,3,2])
            }
            dash_angles = angle_between(vo, ref['vo'], 'byxi, boji->byxoj')

            star0_ = star0[...,None,None,:] * torch.ones_like(ref['po'])[:,None,None]
            star0_under_ref = change_Angle_around_Axis(
                axis * torch.ones_like(star0_),
                star0_,
                ref['po'][:,None,None] * torch.ones_like(star0_),
                0,
                'byxoji ,byxoji ->byxoj '
            )
            po_star0_angles = angle_between(star0_under_ref, ref['po'],  'byxoji, boji->byxoj')

            beta_upper_part = torch.math.cos(dash_angles)
            beta_lower_part = torch.math.cos(po_star0_angles)

            # this uses reduce for min/max, therefore a nan (from division) will be exchanged by the comparison value
            quotient = beta_upper_part / beta_lower_part
            quotient = torch.min(torch.stack([quotient, torch.ones_like(quotient) - 0.0001], dim=-1), dim=-1).values
            quotient = torch.max(torch.stack([quotient, -torch.ones_like(quotient) + 0.0001], dim=-1), dim=-1).values
            beta = torch.acos(quotient)

            R_betas = make_R_from_angle_axis(torch.stack([beta,-beta],axis=-1), axis)

            allp_pos = torch.einsum('byxojaki,byxoji->byxojak', R_betas, star0_under_ref)
            allp_pos = torch.concat([allp_pos[...,0,:], allp_pos[...,1,:]], axis=-2)

            allp_po_angles = angle_between(torch.no_grad(allp_pos), ref['po'], 'byxosi, boji->byxosj')
            allp_angle_diffs = torch.sum(angle_substraction(allp_po_angles, dash_angles.unsqueeze(-2))**2, axis=-1)

            arg_min = torch.argmin(allp_angle_diffs, axis=-1)
            best_po = torch.gather(allp_pos, arg_min.unsqueeze(-1), batch_dims=4)

            o_wide_error = torch.sum(torch.min(allp_angle_diffs, axis=-1)[...,None,:] * iseg[...,None], axis=[1,2], keepdims=True)
            arg_min = torch.argmin(o_wide_error, axis=-1)
            arg_min_ = arg_min * iseg.to(arg_min.dtype)
            best_po = torch.gather(best_po, dim=-2, index=arg_min_.unsqueeze(-1), batch_dims=3)

            return torch.sum(best_po * iseg[..., None], dim=-2)

        if self.model_info["symmetries_continuous"]:
            print("destarring as symmetries_continuous")
            return best_continues_po(postar, torch.tensor([0, 0, 1], dtype=torch.float32))

        if len(self.model_info["symmetries_discrete"]) == 0:
            print("destarring is not changing anything")
            return postar

        if isclose(self.model_info["symmetries_discrete"][0][2,2], 1, abs_tol=1e-3):
            factor = len(self.model_info["symmetries_discrete"])+1
            print("destarring as symmetries_discrete with z_factor=", factor)
            po_ = best_symmetrical_po(get_obj_star0_from_obj_star(postar, z_factor=factor), factor, torch.tensor([0, 0, 1], dtype=torch.float32))

            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            print("po_ was corrected by", -offset)
            return po_ - offset

        if isclose(self.model_info["symmetries_discrete"][0][1,1], 1, abs_tol=1e-3):
            factor = len(self.model_info["symmetries_discrete"])+1
            print("destarring as symmetries_discrete with y_factor=", factor)
            po_ = best_symmetrical_po(get_obj_star0_from_obj_star(postar, y_factor=factor), factor, torch.tensor([0, 1, 0], dtype=torch.float32))

            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            print("po_ was corrected by", -offset)
            return po_ - offset
        
        assert(False)

    def generate_ref_data(self, x: list, counts: int, sample_size: int) -> dict:
        """
        Generate reference data.

        Args:
            x (list): The input list.
            counts (int): The number of counts.
            sample_size (int): The sample size.

        Returns:
            dict: The reference data dictionary.
        """
        postar, vo, iseg = x

        def generate_samples_per_batch(sb_x):
            sb_postar, sb_vo, sb_iseg = sb_x

            def generate_samples_per_instance(i):
                si_seg = sb_iseg[:,:,i]
                selection_index = torch.where(si_seg > 0.5)
                if torch.shape(selection_index)[0] > 0:
                    vo_sel = torch.gather(sb_vo, 0, selection_index)
                    postar_sel = torch.gather(sb_postar, 0, selection_index)

                    pos = torch.randint(0, (vo_sel.size())[0], (counts, sample_size, 1),
                                        dtype=torch.int32)

                    vo_samples = torch.gather(vo_sel, 0, pos)
                    postar_samples = torch.gather(postar_sel, 0, pos)
                else:
                    vo_samples = torch.ones((counts, sample_size, 3))     # a.k.a zeros but this would interprete to zero angle to any other vector
                    postar_samples = torch.ones((counts, sample_size, 3)) # a.k.a zeros but this would interprete to zero angle to any other vector

                return torch.stack([vo_samples, postar_samples], dim=-1)
            return torch.cat([generate_samples_per_instance(i) for i in range(self.amount_of_instances)], dim=0)


        samples = torch.stack([generate_samples_per_batch(sb_x) for sb_x in [postar, vo, iseg]], dim=0)

        vo_samples = samples[...,0]
        postar_samples = samples[...,1]

        def angle_substraction(a,b):
            return torch.minimum(torch.abs(a-b), torch.abs(torch.minimum(a,b) + np.pi * 2. - torch.maximum(a,b)))

        def make_ref_outof_samples_symmetrical(star0_samples, factor, axis):
            ref_vo = vo_samples
            dash_angles = angle_between(vo_samples, ref_vo, dot_product='bski, bsji->bsjk')

            symR = make_R_from_angle_axis(2*np.pi / factor, axis)

            allp_pos = star0_samples[...,None,:]
            for _ in range(factor-1):
                newp = torch.einsum('ij, bskj -> bski', symR, allp_pos[...,-1,:])
                allp_pos = torch.cat([allp_pos, newp[...,None,:]], dim=-2)

            assert(sample_size == 3)
            mg = np.meshgrid(np.arange(factor), np.arange(factor), np.arange(factor))
            gather_per = torch.tensor(np.stack(mg, axis=0).reshape((sample_size,-1)))
            gather_per = gather_per[...,None] *  torch.ones_like(allp_pos[:,:,:,:1,:1], gather_per.dtype)
            gather_per_rev = torch.tensor(np.stack(mg, axis=-1).reshape((-1,sample_size)))
            all_combi = torch.gather(allp_pos, gather_per, dim=3, batch_dims=3) 
            print(all_combi)

            all_combi_po_angles = angle_between(all_combi, all_combi, dot_product='bskni, bsjni->bsjkn')

            all_combi_angle_diffs = torch.sum(angle_substraction(all_combi_po_angles, dash_angles.unsqueeze(-1))**2, dim=[-2,-3])

            arg_min = torch.argmin(all_combi_angle_diffs, dim=-1)

            arg_min_combi = torch.gather(gather_per_rev, dim=0, index=arg_min.unsqueeze(-1), batch_dims=0)

            best_pos = torch.gather(allp_pos, dim=3, index=arg_min_combi.unsqueeze(-1), batch_dims=3) 

            ref_po = best_pos

            return (ref_vo, ref_po)
    
        if self.model_info["symmetries_continuous"]:
            print("generate ref samples for continuous symmetries around z")
            
            ref_vo, ref_po = make_ref_outof_samples_symmetrical(torch.tensor([0,0,1],dtype=torch.float32))

        elif isclose(self.model_info["symmetries_discrete"][0][2,2], 1, abs_tol=1e-3):
            factor = len(self.model_info["symmetries_discrete"])+1
            print("generate ref samples discrete symmetries with z_factor=", factor)
            
            ref_vo, ref_po = make_ref_outof_samples_symmetrical(get_obj_star0_from_obj_star(postar_samples, z_factor=factor),factor, torch.tensor([0,0,1],dtype=torch.float32))

        elif isclose(self.model_info["symmetries_discrete"][0][1,1], 1, abs_tol=1e-3):
            factor = len(self.model_info["symmetries_discrete"])+1
            print("generate ref samples discrete symmetries with y_factor=", factor)
            
            ref_vo, ref_po = make_ref_outof_samples_symmetrical(get_obj_star0_from_obj_star(postar_samples, y_factor=factor), factor, torch.tensor([0,1,0],dtype=torch.float32))
        else:
            assert(False)

        return { 'po': ref_po, 'vo': ref_vo }
