from compression_src.pruning.common import *
from compression_src.reducing.common import *

import torch
from math import *

def yolov8_pruning(model, sparsity):
    # Create a list of (backbone) blocks from the model
    block_list = [model[i] for i in range(23)]
    
    # Iterate through each block in the model
    for i, block in enumerate(block_list):
        # Check if the block is a Conv layer
        if type(block).__name__ == 'Conv':
            # Get the indices of filters to be pruned based on the specified sparsity
            pruning_idx = get_filter_pruning_idx(layer=block.conv, sparsity=sparsity)
            # Perform filter pruning on the convolutional layer using the calculated indices
            filter_pruning(layer=block.conv, pruning_idx=pruning_idx)
            # Prune the corresponding batch normalization layer using the same indices
            bn_pruning(block.bn, pruning_idx=pruning_idx)
        
        # Check if the block is a C2f or SPPF layer
        if type(block).__name__ in ['C2f', 'SPPF']:
            # Get the indices of filters to be pruned based on the specified sparsity
            pruning_idx = get_filter_pruning_idx(layer=block.cv2.conv, sparsity=sparsity)
            # Perform filter pruning on the convolutional layer using the calculated indices
            filter_pruning(layer=block.cv2.conv, pruning_idx=pruning_idx)
            # Prune the corresponding batch normalization layer using the same indices
            bn_pruning(block.cv2.bn, pruning_idx=pruning_idx)

        # Check if the block is a Detect layer
        if type(block).__name__ == 'Detect':
            # Iterate through each set of convolutional layers in the Detect layer
            for i in range(3):
                for j in range(2):
                    # Get the indices of filters to be pruned in cv2 convolutional layers
                    pruning_idx = get_filter_pruning_idx(layer=block.cv2[i][j].conv, sparsity=sparsity)
                    # Perform filter pruning on the cv2 convolutional layers using the calculated indices
                    filter_pruning(layer=block.cv2[i][j].conv, pruning_idx=pruning_idx)
                    # Prune the corresponding batch normalization layers using the same indices
                    bn_pruning(block.cv2[i][j].bn, pruning_idx=pruning_idx)

                    # Get the indices of filters to be pruned in cv3 convolutional layers
                    pruning_idx = get_filter_pruning_idx(layer=block.cv3[i][j].conv, sparsity=sparsity)
                    # Perform filter pruning on the cv3 convolutional layers using the calculated indices
                    filter_pruning(layer=block.cv3[i][j].conv, pruning_idx=pruning_idx)
                    # Prune the corresponding batch normalization layers using the same indices
                    bn_pruning(block.cv3[i][j].bn, pruning_idx=pruning_idx)


def yolov8_reducing(model, reduced_model):
    # Get the indices of filters that survived the first Conv layer
    survived_idx = get_survived_filter_idx(model[0].conv)
    
    # Reduce the first Conv layer using the survived indices
    conv_reduce(
        layer=model[0].conv,
        reduced_layer=reduced_model[0].conv,
        survived_out_channels_idx=survived_idx,
        survived_in_channels_idx=torch.arange(3),  # RGB input channels
    )
    
    # Reduce the corresponding Batch Normalization layer using the same indices
    bn_reduce(model[0].bn, reduced_model[0].bn, survived_idx)
    
    # Initialize previous survived indices for use in subsequent layers
    prev_survived_idx = survived_idx
    
    # Create lists of blocks from the original model and the reduced model
    block_list = [model[i] for i in range(1, 23)]
    reduced_block_list = [reduced_model[i] for i in range(1, 23)]
    
    # Iterate through each block and reduce it
    for i, (block, reduced_block) in enumerate(zip(block_list, reduced_block_list)):
        
        # Check if the current block is a Conv layer
        if type(block).__name__ == 'Conv':
            
            # Get the indices of filters that survived in the current Conv layer
            survived_idx = get_survived_filter_idx(block.conv)

            # Reduce the current Conv layer using the survived indices
            conv_reduce(
                layer=block.conv,
                reduced_layer=reduced_block.conv,
                survived_out_channels_idx=survived_idx,
                survived_in_channels_idx=prev_survived_idx,
            )

            # Reduce the corresponding Batch Normalization layer using the same indices
            bn_reduce(block.bn, reduced_block.bn, survived_idx)
        
        # Check if the current block is a C2f layer
        elif type(block).__name__ in ['C2f', 'SPPF']:
            
            # Handle prev_survived_idx separately for layers where the input is created by concatenating outputs from multiple layers
            if i in [11, 14, 17, 20]: 
                # Define n1 and n2 as the layers whose outputs are concatenated
                n1, n2 = (8, 5) if i == 11 else (11, 3) if i == 14 else (15, 11) if i == 17 else (18, 8) if i == 20 else (None, None)
                
                # Get the survived indices from the first layer
                p_surv_idx_1 = get_survived_filter_idx(block_list[n1].conv if i in [17, 20] else block_list[n1].cv2.conv)
                
                # Determine the number of output channels from the first layer (n1) to adjust the indices accordingly.
                p_surv_idx_2p = block_list[n1].conv.out_channels if i in [17, 20] else block_list[n1].cv2.conv.out_channels
                
                # Get the survived indices from the second layer
                p_surv_idx_2 = get_survived_filter_idx(block_list[n2].cv2.conv) + p_surv_idx_2p
                
                # Concatenate the indices from the first and second layers to create prev_survived_idx (input channels for the current block)
                prev_survived_idx = torch.concat([p_surv_idx_1, p_surv_idx_2])

            # Reduce the cv1.conv layer using the survived indices
            survived_idx = torch.arange(block.cv1.conv.out_channels)
            conv_reduce(
                layer=block.cv1.conv,
                reduced_layer=reduced_block.cv1.conv,
                survived_out_channels_idx=survived_idx,
                survived_in_channels_idx=prev_survived_idx,
            )

            # Prepare indices for the next layer (cv2.con layer)
            prev_survived_idx = torch.arange(block.cv2.conv.in_channels)
            survived_idx = get_survived_filter_idx(block.cv2.conv)

            # Reduce the cv2.conv layer using the survived indices
            conv_reduce(
                layer=block.cv2.conv,
                reduced_layer=reduced_block.cv2.conv,
                survived_out_channels_idx=survived_idx,
                survived_in_channels_idx=prev_survived_idx,
            )

            # Reduce the corresponding Batch Normalization layer using the same indices
            bn_reduce(block.cv2.bn, reduced_block.cv2.bn, survived_idx)

        # Check if the current block is a Detect layer
        elif type(block).__name__ == 'Detect':
            
            # Define the layers that provide input to the current layer and the layers that need to be reduced
            prev_indices = [14, 17, 20] # Layers that connect to the current layer
            cv_layers = [0, 1, 2] # Layers in the Detect block to be reduced

            # Iterate through each set of cv2 and cv3 layers in the Detect layer
            for idx, prev_idx in zip(cv_layers, prev_indices):
                
                for sub_layer, reduced_sub_layer in zip([block.cv2, block.cv3], [reduced_block.cv2, reduced_block.cv3]):
                    
                    # Get the survived indices from the previous layers
                    prev_survived_idx = get_survived_filter_idx(block_list[prev_idx].cv2.conv)

                    for i in range(2):
                        # Get the survived indices for the current layer and reduce it
                        survived_idx = get_survived_filter_idx(sub_layer[idx][i].conv)
                        conv_reduce(
                            layer=sub_layer[idx][i].conv,
                            reduced_layer=reduced_sub_layer[idx][i].conv,
                            survived_out_channels_idx=survived_idx,
                            survived_in_channels_idx=prev_survived_idx,
                        )
                        bn_reduce(sub_layer[idx][i].bn, reduced_sub_layer[idx][i].bn, survived_idx)
                        prev_survived_idx = survived_idx

                    # For the Conv2d layer, only the corresponding input channels are reduced
                    # The output channels of Conv2d are pre-defined by self.reg_max and self.nc, so reducing them incorrectly may cause errors
                    survived_idx = get_survived_filter_idx(sub_layer[idx][2])
                    conv_reduce(
                        layer=sub_layer[idx][2],
                        reduced_layer=reduced_sub_layer[idx][2],
                        survived_out_channels_idx=survived_idx,
                        survived_in_channels_idx=prev_survived_idx,
                    )

        # Update prev_survived_idx for the next block
        prev_survived_idx = survived_idx