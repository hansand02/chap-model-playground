"""TensorBoard visualization utilities for training."""

from torch.utils.tensorboard import SummaryWriter


def create_writer(name=None):
    """Create a TensorBoard writer."""
    return SummaryWriter(comment=f"_{name}" if name else "")


def log_loss(writer, loss, epoch):
    """Log training loss."""
    writer.add_scalar("Loss/train", loss, epoch)


def log_metrics(writer, mse, mae, epoch):
    """Log evaluation metrics."""
    writer.add_scalars("Metrics", {"MSE": mse, "MAE": mae}, epoch)


def log_model(writer, model, input_tensor):
    """Log model architecture graph."""
    writer.add_graph(model, input_tensor)
