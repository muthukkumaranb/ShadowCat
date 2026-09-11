import pytest
import torch

from lstm.model import LSTMClassifier


def test_model_output_shape():
    batch_size = 8
    sequence_length = 30
    num_features = 12

    model = LSTMClassifier(
        input_size=num_features,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        num_features,
    )

    output = model(x)

    assert output.shape == (batch_size,)


def test_probability_shape():
    batch_size = 8
    sequence_length = 30
    num_features = 12

    model = LSTMClassifier(
        input_size=num_features,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        num_features,
    )

    probabilities = model.predict_proba(x)

    assert probabilities.shape == (batch_size,)


def test_logits_match_probability_output():
    model = LSTMClassifier(input_size=5, dropout=0.0)
    x = torch.randn(4, 10, 5)
    logits = model.forward_logits(x)
    probabilities = model(x)
    assert logits.shape == (4,)
    assert torch.allclose(torch.sigmoid(logits), probabilities)


def test_probability_range():
    model = LSTMClassifier(
        input_size=5,
    )

    x = torch.randn(
        16,
        20,
        5,
    )

    probabilities = model.predict_proba(x)

    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_different_feature_counts():
    model_5_features = LSTMClassifier(
        input_size=5,
    )

    model_17_features = LSTMClassifier(
        input_size=17,
    )

    x1 = torch.randn(4, 30, 5)
    x2 = torch.randn(4, 30, 17)

    y1 = model_5_features(x1)
    y2 = model_17_features(x2)

    assert y1.shape == (4,)
    assert y2.shape == (4,)


def test_invalid_input_dimension():
    model = LSTMClassifier(
        input_size=10,
    )

    x = torch.randn(8, 10)

    with pytest.raises(ValueError):
        model(x)


def test_wrong_feature_count():
    model = LSTMClassifier(
        input_size=10,
    )

    x = torch.randn(
        8,
        30,
        7,
    )

    with pytest.raises(ValueError):
        model(x)


def test_invalid_model_configuration():
    with pytest.raises(ValueError):
        LSTMClassifier(input_size=0)

    with pytest.raises(ValueError):
        LSTMClassifier(
            input_size=10,
            hidden_size=0,
        )

    with pytest.raises(ValueError):
        LSTMClassifier(
            input_size=10,
            num_layers=0,
        )

    with pytest.raises(ValueError):
        LSTMClassifier(
            input_size=10,
            dropout=1.0,
        )