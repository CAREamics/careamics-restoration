"""Base model for the various CAREamics architectures."""

from typing import Any, Dict

from pydantic import BaseModel


class ArchitectureModel(BaseModel):
    """
    Base Pydantic model for all model architectures.

    The `model_dump` method allows removing the `architecture` key from the model.
    """

    architecture: str
    """Name of the architecture."""

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Dump the model as a dictionary, ignoring the architecture keyword.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments from Pydantic BaseModel model_dump method.

        Returns
        -------
        dict[str, Any]
            Model as a dictionary.
        """
        model_dict = super().model_dump(**kwargs)

        # remove the architecture key
        model_dict.pop("architecture")

        return model_dict
