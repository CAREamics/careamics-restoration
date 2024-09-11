import pytest

from careamics.config.training_model import TrainingConfig


def test_training_wrong_values_by_assignments(minimum_training: dict):
    """Test that wrong values cause an error during assignment."""
    training = TrainingConfig(**minimum_training)

    # num_epochs
    training.num_epochs = 2
    with pytest.raises(ValueError):
        training.num_epochs = -1

    training.max_steps = 2
    with pytest.raises(ValueError):
        training.max_steps = 0

    training.precision = "16-mixed"
    with pytest.raises(ValueError):
        training.precision = "16"
