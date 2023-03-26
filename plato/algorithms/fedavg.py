"""
The federated averaging algorithm for PyTorch.
"""
from plato.algorithms import base


class Algorithm(base.Algorithm):
    """PyTorch-based federated averaging algorithm, used by both the client and the server."""
    def extract_weights(self):
        """Extract weights from the model."""
        import logging
        logging.info("888888888888888888888888888888888888888888888888888888")
        return self.model.cpu().state_dict()

    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""
        import logging
        logging.info("999999999999999999999999999999999999999999999999999999")
        self.model.load_state_dict(weights, strict=True)
