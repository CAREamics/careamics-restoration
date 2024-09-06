import numpy as np
from torch import Tensor

from careamics import CAREamist
from careamics.model_io import export_to_bmz, load_pretrained
from careamics.model_io.bmz_io import _export_state_dict, _load_state_dict


def test_state_dict_io(tmp_path, ordered_array, pre_trained):
    """Test exporting and loading a state dict."""
    # training data
    train_array = ordered_array((32, 32))
    path = tmp_path / "model.pth"

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # predict (no tiling and no tta)
    predicted_output = careamist.predict(train_array, tta_transforms=False)
    predicted = np.concatenate(predicted_output, axis=0)

    # save model
    _export_state_dict(careamist.model, path)
    assert path.exists()

    # load model
    _load_state_dict(careamist.model, path)

    # predict (no tiling and no tta)
    predicted_loaded = careamist.predict(train_array, tta_transforms=False)
    assert (predicted_loaded == predicted).all()


def test_bmz_io(tmp_path, ordered_array, pre_trained):
    """Test exporting and loading to the BMZ."""
    # training data
    train_array = ordered_array((32, 32))

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # predict (no tiling and no tta)
    predicted_output = careamist.predict(train_array, tta_transforms=False)
    predicted = np.concatenate(predicted_output, axis=0)

    # export to BioImage Model Zoo
    path = tmp_path / "other_folder" / "model.zip"
    export_to_bmz(
        model=careamist.model,
        config=careamist.cfg,
        path_to_archive=path,
        model_name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
        input_array=train_array[np.newaxis, np.newaxis, ...],
        output_array=predicted,
    )
    assert path.exists()

    # load model
    model, config = load_pretrained(path)
    assert config == careamist.cfg

    # compare predictions
    torch_array = Tensor(train_array[np.newaxis, np.newaxis, ...])
    predicted = careamist.model.forward(torch_array).detach().numpy().squeeze()
    predicted_loaded = model.forward(torch_array).detach().numpy().squeeze()
    assert (predicted_loaded == predicted).all()
