import torch


class Device:
    """Device that training and prediction run on, with the settings that differ between devices."""

    def __init__(self, name: str | None = None, amp: bool = False, benchmark: bool = False) -> None:
        """Select the device and set up its mixed precision and cudnn.

        Args:
            name: torch device such as cuda, cuda:1 or cpu, None for cuda when available and cpu otherwise.
            amp: mixed precision on cuda, bfloat16 if supported and float16 with gradient scaling otherwise.
            benchmark: let cudnn pick the fastest conv for each input size, for training with few input sizes.
        """
        if name is not None and name.split(":")[0] not in ["cuda", "cpu"]:
            raise SystemExit(f"unknown device {name}, expected cuda, cuda:<index> or cpu")

        self.torch_device = torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
        is_cuda = self.torch_device.type == "cuda"

        if is_cuda and not torch.cuda.is_available():
            raise SystemExit(f"device {name} is not available, use cpu")

        if is_cuda and self.torch_device.index is not None and self.torch_device.index >= torch.cuda.device_count():
            raise SystemExit(f"device {name} is not available, there are {torch.cuda.device_count()} cuda devices")

        # the operations without an explicit device then run on this one too
        if is_cuda and self.torch_device.index is not None:
            torch.cuda.set_device(self.torch_device.index)

        torch.backends.cudnn.benchmark = benchmark and is_cuda
        self.pin_memory = is_cuda

        # bfloat16 needs no gradient scaling, and a disabled scaler passes everything through
        self.amp_dtype = None

        if amp and is_cuda:
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.scaler = torch.amp.GradScaler(self.torch_device.type, enabled=self.amp_dtype == torch.float16)

    def autocast(self) -> torch.autocast:
        """Return the mixed precision context of the device, which does nothing when AMP is off."""
        return torch.autocast(self.torch_device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None)

    def state(self) -> dict:
        """Return the random state of the gpu and the gradient scaler state, to save in a checkpoint."""
        random_state = torch.cuda.get_rng_state(self.torch_device) if self.torch_device.type == "cuda" else None
        return {"random": random_state, "scaler": self.scaler.state_dict()}

    def restore(self, state: dict) -> None:
        """Restore a state from state(), skipping the parts that do not apply to this device."""
        # a gpu random state applies only to a gpu
        if state["random"] is not None and self.torch_device.type == "cuda":
            torch.cuda.set_rng_state(state["random"], self.torch_device)

        # the scaler state is empty when it was saved with bfloat16 or on the cpu
        if state["scaler"]:
            self.scaler.load_state_dict(state["scaler"])

    @staticmethod
    def list_devices() -> list[tuple[str, str]]:
        """List the available devices as (name, description), the cuda devices first and then the cpu."""
        cuda_devices = [(f"cuda:{i}", torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
        return [*cuda_devices, ("cpu", "CPU")]
