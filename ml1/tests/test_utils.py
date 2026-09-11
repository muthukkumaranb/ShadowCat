from pathlib import Path

import numpy as np
import pytest
import torch

from lstm.model import LSTMClassifier
from lstm.utils import (
    count_parameters,
    get_device,
    load_model,
    model_summary,
    save_model,
    set_seed,
)


def create_model(
    features: int = 5,
) -> LSTMClassifier:
    return LSTMClassifier(
        input_size=features,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )


def test_set_seed_reproducibility():

    set_seed(42)

    numpy_values_1 = np.random.randn(5)
    torch_values_1 = torch.randn(5)

    set_seed(42)

    numpy_values_2 = np.random.randn(5)
    torch_values_2 = torch.randn(5)

    assert np.array_equal(
        numpy_values_1,
        numpy_values_2,
    )

    assert torch.equal(
        torch_values_1,
        torch_values_2,
    )


def test_invalid_seed():

    with pytest.raises(TypeError):
        set_seed("42")


def test_get_device_cpu():

    device = get_device("cpu")

    assert isinstance(
        device,
        torch.device,
    )

    assert device.type == "cpu"


def test_get_device_auto():

    device = get_device()

    assert isinstance(
        device,
        torch.device,
    )

    assert device.type in {
        "cpu",
        "cuda",
    }


def test_invalid_device():

    with pytest.raises(ValueError):
        get_device("invalid_device")


def test_count_parameters():

    model = create_model()

    parameter_count = count_parameters(
        model
    )

    assert isinstance(
        parameter_count,
        int,
    )

    assert parameter_count > 0


def test_model_summary():

    model = create_model()

    summary = model_summary(
        model
    )

    assert summary["model_class"] == (
        "LSTMClassifier"
    )

    assert summary["trainable_parameters"] > 0

    assert summary["device"] in {
        "cpu",
        "cuda",
    }

    assert summary["training_mode"] is True


def test_save_model(tmp_path):

    model = create_model()

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    returned_path = save_model(
        model,
        checkpoint_path,
    )

    assert returned_path == checkpoint_path
    assert checkpoint_path.exists()


def test_save_model_creates_parent_directory(
    tmp_path,
):

    model = create_model()

    checkpoint_path = (
        tmp_path
        / "artifacts"
        / "models"
        / "model.pt"
    )

    save_model(
        model,
        checkpoint_path,
    )

    assert checkpoint_path.exists()


def test_save_model_with_metadata(
    tmp_path,
):

    model = create_model()

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    metadata = {
        "version": "v1",
        "input_size": 5,
    }

    save_model(
        model,
        checkpoint_path,
        metadata=metadata,
    )

    loaded_model, loaded_metadata = (
        load_model(
            create_model(),
            checkpoint_path,
            device="cpu",
        )
    )

    assert loaded_metadata == metadata


def test_load_model(
    tmp_path,
):

    model = create_model()

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    save_model(
        model,
        checkpoint_path,
    )

    loaded_model, metadata = load_model(
        create_model(),
        checkpoint_path,
        device="cpu",
    )

    assert isinstance(
        loaded_model,
        LSTMClassifier,
    )

    assert metadata == {}


def test_loaded_model_has_same_weights(
    tmp_path,
):

    model = create_model()

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    save_model(
        model,
        checkpoint_path,
    )

    loaded_model, _ = load_model(
        create_model(),
        checkpoint_path,
        device="cpu",
    )

    original_state = (
        model.state_dict()
    )

    loaded_state = (
        loaded_model.state_dict()
    )

    assert original_state.keys() == (
        loaded_state.keys()
    )

    for key in original_state:

        assert torch.equal(
            original_state[key],
            loaded_state[key],
        )


def test_loaded_model_produces_same_output(
    tmp_path,
):

    model = create_model()

    model.eval()

    X = torch.randn(
        4,
        10,
        5,
    )

    with torch.no_grad():
        original_output = model(X)

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    save_model(
        model,
        checkpoint_path,
    )

    loaded_model, _ = load_model(
        create_model(),
        checkpoint_path,
        device="cpu",
    )

    loaded_model.eval()

    with torch.no_grad():
        loaded_output = loaded_model(X)

    assert torch.allclose(
        original_output,
        loaded_output,
        atol=1e-6,
    )


def test_missing_model_file(
    tmp_path,
):

    model = create_model()

    missing_path = (
        tmp_path / "missing.pt"
    )

    with pytest.raises(FileNotFoundError):

        load_model(
            model,
            missing_path,
        )


def test_invalid_checkpoint(
    tmp_path,
):

    invalid_path = (
        tmp_path / "invalid.pt"
    )

    torch.save(
        {"wrong_key": {}},
        invalid_path,
    )

    with pytest.raises(ValueError):

        load_model(
            create_model(),
            invalid_path,
            device="cpu",
        )