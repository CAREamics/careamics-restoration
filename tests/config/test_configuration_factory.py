import pytest

from careamics.config import (
    create_care_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
)
from careamics.config.configuration_factory import (
    _create_supervised_configuration,
    _create_unet,
)
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
    SupportedTransform,
)
from careamics.config.transformations import (
    N2VManipulateModel,
    XYFlipModel,
    XYRandomRotate90Model,
)


def test_model_creation():
    """Test that the correct parameters are passed to the model."""
    model_kwargs = {
        "depth": 4,
        "conv_dims": 2,
        "n2v2": False,
        "in_channels": 2,
        "num_classes": 5,
        "independent_channels": False,
    }

    # choose different parameters
    axes = "XYZ"
    conv_dims = 3
    in_channels = 3
    num_classes = 4
    independent_channels = True
    use_n2v2 = True

    model = _create_unet(
        axes=axes,
        n_channels_in=in_channels,
        n_channels_out=num_classes,
        independent_channels=independent_channels,
        use_n2v2=use_n2v2,
        model_kwargs=model_kwargs,
    )

    assert model.depth == model_kwargs["depth"]
    assert model.conv_dims == conv_dims
    assert model.n2v2 == use_n2v2
    assert model.in_channels == in_channels
    assert model.num_classes == num_classes
    assert model.independent_channels == independent_channels


def test_supervised_configuration_error_with_channel_axes():
    """Test that an error is raised if channels are in axes, but the input channel
    number is not specified."""
    with pytest.raises(ValueError):
        _create_supervised_configuration(
            algorithm="n2n",
            experiment_name="test",
            data_type="tiff",
            axes="CYX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )


def test_supervised_configuration_error_without_channel_axes():
    """Test that an error is raised if channels are not in axes, but the input channel
    number is specified and greater than 1."""
    with pytest.raises(ValueError):
        _create_supervised_configuration(
            algorithm="n2n",
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
            n_channels_in=2,
        )


def test_supervised_configuration_channels():
    """Test that no error is raised if channels are in axes and the input channel
    are specified."""
    _create_supervised_configuration(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=4,
    )


def test_supervised_configuration_aug():
    """Test that the default augmentations are present."""
    config = _create_supervised_configuration(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert len(config.data_config.transforms) == 2
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value
    assert (
        config.data_config.transforms[1].name
        == SupportedTransform.XY_RANDOM_ROTATE90.value
    )


def test_supervised_configuration_no_aug():
    """Test that disabling augmentation results in empty transform list."""
    config = _create_supervised_configuration(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_augmentations=False,
    )
    assert len(config.data_config.transforms) == 0


def test_supervised_configuration_no_aug_with_transforms_passed():
    """Test that the augmentation flag also disables passing transforms."""
    config = _create_supervised_configuration(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_augmentations=False,
        augmentations=[XYFlipModel()],
    )
    assert len(config.data_config.transforms) == 0


def test_supervised_configuration_passing_transforms():
    """Test that transforms can be passed to the configuration."""
    config = _create_supervised_configuration(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        augmentations=[XYFlipModel()],
    )
    assert len(config.data_config.transforms) == 1
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value


def test_n2n_configuration():
    """Test that N2N configuration can be created."""
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.algorithm_config.algorithm == "n2n"


def test_n2n_configuration_n_channels():
    """Test the behaviour of the number of channels in and out."""
    n_channels_in = 4
    n_channels_out = 5

    # n_channels_out not specified
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=n_channels_in,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_in

    # specify n_channels_out
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_out


def test_care_configuration():
    """Test that CARE configuration can be created."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.algorithm_config.algorithm == "care"


def test_care_configuration_n_channels():
    """Test the behaviour of the number of channels in and out."""
    n_channels_in = 4
    n_channels_out = 5

    # n_channels_out not specified
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=n_channels_in,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_in

    # specify n_channels_out
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_out


def test_n2v_configuration():
    """Test that N2V configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.algorithm_config.algorithm == "n2v"
    assert config.algorithm_config.loss == "n2v"
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )


def test_n2v_3d_configuration():
    """Test that N2V configuration can be created in 3D."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="ZYX",
        patch_size=[64, 64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )
    assert (
        config.data_config.transforms[-1].strategy
        == SupportedPixelManipulation.UNIFORM.value
    )
    assert config.algorithm_config.model.is_3D()


def test_n2v_configuration_default_transforms():
    """Test the default n2v transforms."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert len(config.data_config.transforms) == 3
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value
    assert (
        config.data_config.transforms[1].name
        == SupportedTransform.XY_RANDOM_ROTATE90.value
    )
    assert (
        config.data_config.transforms[2].name == SupportedTransform.N2V_MANIPULATE.value
    )


def test_n2v_configuration_no_aug():
    """Test that the N2V configuration can be created without augmentation."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_augmentations=False,
    )
    assert len(config.data_config.transforms) == 1  # only N2V manipulate
    assert (
        config.data_config.transforms[0].name == SupportedTransform.N2V_MANIPULATE.value
    )


def test_n2v_configuration_no_aug_with_transforms_passed():
    """Test that the augmentation flag also disables passing transforms."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_augmentations=False,
        augmentations=[XYFlipModel(), XYRandomRotate90Model()],
    )
    assert len(config.data_config.transforms) == 1  # only N2V manipulate
    assert (
        config.data_config.transforms[0].name == SupportedTransform.N2V_MANIPULATE.value
    )


def test_n2v_configuration_passing_n2v_manipulate():
    """Test that N2V manipulate is ignored when explicitely passed."""
    use_n2v2 = True
    roi_size = 15
    masked_pixel_percentage = 0.5
    struct_mask_axis = SupportedStructAxis.HORIZONTAL.value
    struct_n2v_span = 15

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        augmentations=[
            XYFlipModel(),
            N2VManipulateModel(
                strategy=SupportedPixelManipulation.UNIFORM.value,
                roi_size=5,
                masked_pixel_percentage=0.7,
                struct_mask_axis=SupportedStructAxis.VERTICAL.value,
                struct_mask_span=7,
            ),
        ],
        use_n2v2=use_n2v2,  # median strategy
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        struct_n2v_axis=struct_mask_axis,
        struct_n2v_span=struct_n2v_span,
    )
    assert len(config.data_config.transforms) == 2
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value
    assert (
        config.data_config.transforms[1].name == SupportedTransform.N2V_MANIPULATE.value
    )
    assert (
        config.data_config.transforms[1].strategy
        == SupportedPixelManipulation.MEDIAN.value
    )
    assert config.data_config.transforms[1].roi_size == roi_size
    assert (
        config.data_config.transforms[1].masked_pixel_percentage
        == masked_pixel_percentage
    )
    assert config.data_config.transforms[1].struct_mask_axis == struct_mask_axis
    assert config.data_config.transforms[1].struct_mask_span == struct_n2v_span
