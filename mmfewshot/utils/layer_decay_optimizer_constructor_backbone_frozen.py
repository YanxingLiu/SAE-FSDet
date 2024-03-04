# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Mostly copy-paste from BEiT library:

https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/layer_decay_optimizer_constructor.py
"""

import json

from mmcv.runner import (OPTIMIZER_BUILDERS, DefaultOptimizerConstructor,
                         get_dist_info)


def get_num_layer_for_mae(var_name, num_encoder_layers, num_decoder_layers,
                          num_max_layer):
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.blocks'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    elif (var_name.startswith('roi_head.bbox_head.norm')
          or var_name.startswith('roi_head.bbox_head.decoder_embed')):
        return num_encoder_layers + 1
    elif var_name.startswith('roi_head.bbox_head.decoder_blocks'):
        layer_id = int(var_name.split('.')[3])
        return num_encoder_layers + layer_id + 2
    elif var_name in ('roi_head.bbox_head.det_token', ):
        return num_encoder_layers + 1
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructorBackboneFronzen(DefaultOptimizerConstructor
                                                    ):

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print(self.paramwise_cfg)
        num_encoder_layers = self.paramwise_cfg.get('num_encoder_layers')
        num_decoder_layers = self.paramwise_cfg.get('num_decoder_layers')
        num_max_layer = num_encoder_layers + num_decoder_layers + 3
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        print('Build LayerDecayOptimizerConstructor %f - %d' %
              (layer_decay_rate, num_max_layer))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('backbone.'):
                continue
            if (len(param.shape) == 1 or name.endswith('.bias')
                    or name.endswith('_token') or name.endswith('pos_embed')):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_mae(name, num_encoder_layers,
                                             num_decoder_layers, num_max_layer)
            group_name = 'layer_%d_%s' % (layer_id, group_name)
            '''
            # setting backbone frozen
            if name.startswith("backbone."):
                group_name = "backbone"
                parameter_groups[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "param_names": [],
                "lr_scale": scale,
                "group_name": group_name,
                "lr": 0,
                }
            # setting rpn frozen
            elif name.startswith("rpn_head."):
                group_name = "rpn_head"
                parameter_groups[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "param_names": [],
                "lr_scale": scale,
                "group_name": group_name,
                "lr": 0,
                }
            '''

            if group_name not in parameter_groups:
                scale = layer_decay_rate**(num_max_layer - layer_id - 1)
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print('Param groups = %s' % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])
        params.extend(parameter_groups.values())
