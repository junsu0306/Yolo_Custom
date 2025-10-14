import torch


def get_survived_filter_idx(layer):
    weight = layer.weight
    num_filters = weight.shape[0]
    filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
    survived_filter_idx = torch.where(filter_norms != 0)[0]
    return survived_filter_idx


def conv_reduce(
    layer, reduced_layer, survived_out_channels_idx, survived_in_channels_idx
):

    # set reduced_layer's in_channels and out_channels
    reduced_layer.in_channels = len(survived_in_channels_idx)
    reduced_layer.out_channels = len(survived_out_channels_idx)

    # in case of depthwise conv, the number of groups and out_channels have to be same
    if reduced_layer.groups != 1:
        reduced_layer.groups = reduced_layer.out_channels

    # re-defining the weight parameter of reduced_layer
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.out_channels,
            reduced_layer.in_channels,
            reduced_layer.kernel_size[0],
            reduced_layer.kernel_size[1],
        ),
        requires_grad=True,
    )

    weight = layer.weight
    reduced_weight = reduced_layer.weight
    
    assert (
        len(survived_out_channels_idx) == reduced_weight.shape[0]
    ), "reduced_num_filters != survived_idx {}, {}".format(
        len(survived_out_channels_idx), reduced_weight.shape[0]
    )
    
    assert (
        len(survived_in_channels_idx) == reduced_weight.shape[1]
    ), "reduced_num_channel != survived_idx {}, {}".format(
        len(survived_in_channels_idx), reduced_weight.shape[1]
    )

    # copy the survived weights to the reduced layer
    with torch.no_grad():
        reduced_weight.copy_(
            weight[survived_out_channels_idx, :, :, :][
                :, survived_in_channels_idx, :, :
            ]
        )


def fc_reduce(
    layer, reduced_layer, survived_out_features_idx, survived_in_features_idx
):

    # set reduced_layer's in_channels and out_channels
    reduced_layer.in_features = len(survived_in_features_idx)
    reduced_layer.out_features = len(survived_out_features_idx)

    # re-defining the weight parameter of reduced_layer
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.out_features,
            reduced_layer.in_features,
        ),
        requires_grad=True,
    )

    reduced_layer.bias = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.out_features,
        ),
        requires_grad=True,
    )

    weight = layer.weight
    reduced_weight = reduced_layer.weight

    bias = layer.bias
    reduced_bias = reduced_layer.bias

    assert (
        len(survived_out_features_idx) == reduced_weight.shape[0]
    ), "reduced_num_filters != survived_idx {}, {}".format(
        len(survived_out_features_idx), reduced_weight.shape[0]
    )
    assert (
        len(survived_in_features_idx) == reduced_weight.shape[1]
    ), "reduced_num_channel != survived_idx {}, {}".format(
        len(survived_in_features_idx), reduced_weight.shape[1]
    )

    with torch.no_grad():
        reduced_weight.copy_(
            weight[survived_out_features_idx, :][:, survived_in_features_idx]
        )
        reduced_bias.copy_(bias[survived_out_features_idx])


def bn_reduce(layer, reduced_layer, survived_features_idx):
    reduced_layer.num_features = len(survived_features_idx)

    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.num_features,
        ),
        requires_grad=True,
    )
    reduced_layer.bias = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.num_features,
        ),
        requires_grad=True,
    )
    reduced_layer.running_mean = torch.zeros(
        reduced_layer.num_features,
    )
    reduced_layer.running_var = torch.zeros(
        reduced_layer.num_features,
    )

    weight = layer.weight
    reduced_weight = reduced_layer.weight
    bias = layer.bias
    reduced_bias = reduced_layer.bias
    mean = layer.running_mean
    reduced_mean = reduced_layer.running_mean
    var = layer.running_var
    reduced_var = reduced_layer.running_var

    assert (
        len(survived_features_idx) == reduced_weight.shape[0]
    ), "reduced_bn_weight != survived_idx"
    assert (
        len(survived_features_idx) == reduced_bias.shape[0]
    ), "reduced_bn_bias != survived_idx"
    assert (
        len(survived_features_idx) == reduced_mean.shape[0]
    ), "reduced_bn_mean != survived_idx"
    assert (
        len(survived_features_idx) == reduced_var.shape[0]
    ), "reduced_bn_var != survived_idx"
    with torch.no_grad():
        reduced_weight.copy_(weight[survived_features_idx])
        reduced_bias.copy_(bias[survived_features_idx])
        reduced_mean.copy_(mean[survived_features_idx])
        reduced_var.copy_(var[survived_features_idx])


def copy_layer(layer, reduced_layer):
    with torch.no_grad():
        for p, reduced_p in zip(layer.parameters(), reduced_layer.parameters()):
            reduced_p.copy_(p)


def bn_copy_layer(layer, reduced_layer):

    with torch.no_grad():

        mean = layer.running_mean
        reduced_mean = reduced_layer.running_mean
        var = layer.running_var
        reduced_var = reduced_layer.running_var
        reduced_mean.copy_(mean)
        reduced_var.copy_(var)

        for p, reduced_p in zip(layer.parameters(), reduced_layer.parameters()):
            reduced_p.copy_(p)
