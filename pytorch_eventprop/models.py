from typing import Any, Tuple

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch.functional.loss import SpikeTime
from torch.autograd import Function

try:
    from torchmeta.modules import MetaModule
except ImportError:
    from torch.nn import Module as MetaModule


def reset(net):
    for m in net.modules():
        if hasattr(m, "reset_hidden"):
            m.reset_hidden()
        # else:
        #     reset(m)


def init_weights(w, weight_scale=1.0):
    k = weight_scale * (1.0 / np.sqrt(w.shape[1]))
    nn.init.uniform_(w, -k, k)
    print(k)


class RecordingSequential(nn.Sequential):
    def __init__(self, *modules):
        super(RecordingSequential, self).__init__(*modules)

    def forward(self, x, params=None):
        recs = []
        if params is not None:
            # print("Using params")
            assert len(params) == len(self._modules)
        else:
            params = {n: None for n in self._modules}
        for module, param in zip(self._modules.values(), params.values()):
            try:
                x = module(x, params=param)
            except TypeError:
                x = module(x)
            if isinstance(x, tuple):
                recs.append(x[1])
                x = x[0]
            elif isinstance(x, dict):
                if "output" in x:
                    x = x["output"]
                elif "spikes" in x:
                    x = x["spikes"]
                else:
                    raise ValueError(f"Invalid dict {x}")
                recs.append(x)

        try:
            recs = {k: [r[k] for r in recs] for k in recs[0].keys()}
        except TypeError:
            pass
        return {"output": x, "recordings": recs}


class SpikingLinear_ev(MetaModule):
    def __init__(self, d1, d2, **kwargs):
        super(SpikingLinear_ev, self).__init__()

        self.input_dim = d1
        self.output_dim = d2
        self.T = kwargs.get("T")
        self.dt = kwargs.get("dt", 1e-3)
        self.tau_m = kwargs.get("tau_m", 20e-3)
        self.tau_s = kwargs.get("tau_s", 5e-3)
        self.resolve_silent = kwargs.get("resolve_silent", False)

        self.mu = kwargs.get("mu", 0.1)
        self.sigma = kwargs.get("sigma", 0.1)

        self.scale = kwargs.get("scale", 1.0)
        self.mu_silent = 1 / np.sqrt(d1)
        self.seed = kwargs.get("seed", None)

        self.dropout_p = kwargs.get("dropout", None)

        # self.alpha = 1 - self.dt / self.tau_s
        # self.beta = 1 - self.dt / self.tau_m

        self.alpha = np.exp(-self.dt / self.tau_s)
        self.beta = np.exp(-self.dt / self.tau_m)

        self.max_delay = int(kwargs.get("max_delay") // 1) #think should be dt, bit confused about the timestep in the code

        if self.max_delay > 0:
            self.P = nn.Parameter(torch.Tensor(d2, d1),requires_grad=False)
            self.IDX = torch.nn.Parameter(torch.arange(0, self.max_delay),requires_grad=False).expand(d2, d1, -1).permute(2, 0, 1)
        self.weight = nn.Parameter(torch.Tensor(d2, d1))

        self.init_mode = kwargs.get("init_mode", "kaiming")

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        if self.init_mode == "kaiming":
            nn.init.kaiming_normal_(self.weight)
        elif self.init_mode == "kaiming_both":
            div = 1 / np.sqrt(nn.init._calculate_fan_in_and_fan_out(self.weight)[0])
            if isinstance(self.scale, list):
                mu, sigma = [self.scale[0] * div, self.scale[1] * div]
            else:
                mu, sigma = self.scale * div, self.scale * div
            self.weight.data = torch.from_numpy(
                np.random.normal(mu, sigma, self.weight.T.shape).T
            ).float()
            # nn.init.normal_(self.weight, mu, mu)
        elif self.init_mode == "normal" or "paper" in self.init_mode:
            nn.init.normal_(self.weight, self.mu, self.sigma)
        elif self.init_mode == "uniform":
            nn.init.uniform_(self.weight, -self.mu, self.mu)
        else:
            raise ValueError(f"Invalid init_mode {self.init_mode}")
        
        if self.max_delay > 0:
            nn.init.uniform_(self.P, 0, self.max_delay)
            self.P.round_()

        if self.init_mode != "kaiming_both" and not isinstance(self.scale, list):
            self.weight.data *= self.scale

        self.reset_to_zero = kwargs.get("reset_to_zero", False)

        self.device = kwargs.get(
            "device",
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )

        # self.forward = lambda input: self.EventProp.apply(
        #     input, self.weight, self.manual_forward, self.manual_backward
        # )

        # Only store a limited number of backward dictionaries to prevent memory growth
        self.max_bwd_dicts = kwargs.get("max_bwd_dicts", 1)
        self.bwd_dicts = []

    def forward(self, input, params=None):
        if params is not None:
            # print("Layer Using params")
            return self.EventProp.apply(input, params, self.manual_forward, self.manual_backward)
        else:
            return self.EventProp.apply(
                input, self.weight, self.manual_forward, self.manual_backward
            )

    @staticmethod
    class EventProp(Function):
        @staticmethod
        def forward(input, weights, manual_forward, manual_backward):
            output, pack, fwd_dict = manual_forward(input, weights)
            return output, fwd_dict, pack

        @staticmethod
        def setup_context(ctx: Any, inputs: Tuple[Any], outputs: Any) -> Any:
            input, weights, manual_forward, manual_backward = inputs
            *_, pack = outputs

            ctx.backward = manual_backward
            ctx.weights = weights
            ctx.save_for_backward(*pack)

        @staticmethod
        def backward(ctx, *grad_output):
            backward = ctx.backward
            pack = ctx.saved_tensors
            weights = ctx.weights

            # Call the manual backward function which adds to bwd_dicts internally
            grad_input, grad_weights, lV, lI = backward(grad_output[0], pack, weights)

            # Create backward dictionary to return (though not used directly)
            bwd_dict = {
                "grad_input": grad_input,
                "grad_weights": grad_weights,
                "lV": lV,
                "lI": lI,
            }

            return grad_input, grad_weights, None, None

    def __repr__(self):
        return super().__repr__()[:-1] + f"{self.input_dim}, {self.output_dim})"

    def manual_forward(self, input, weights=None):
        steps = input.shape[0]
        assert self.T == steps, f"Input steps {steps} != T {self.T}"

        V = torch.zeros(steps, input.shape[1], self.output_dim).to(self.device)
        I = torch.zeros(steps, input.shape[1], self.output_dim).to(self.device)
        V_spikes = torch.zeros(steps, input.shape[1], self.output_dim).to(self.device)
        output = torch.zeros(steps, input.shape[1], self.output_dim).to(self.device)

        # input = torch.roll(input, 1, dims=0)
        # print("Spike is at time", input.argmax().data.item())
        if self.max_delay > 0:
            X = self.IDX - self.P
            X = ((1 - X.abs()).relu())
            K = (X * self.weight)
            K = K.permute(1, 2, 0)
            input_padded = F.pad(input.permute(1,2,0), (self.max_delay-1, 0), 'constant', 0)

        while True:
            for i in range(1, steps):
                # spikes = (V[i - 1] > 1.0).float()
                # V[i - 1] = (1 - spikes) * V[i - 1]
                if self.max_delay > 0:
                    input_t = F.conv1d(
                        input_padded[:,:,i - 1:i - 1 + self.max_delay].float(),
                        K,
                    ).permute(2,0,1)
                else:
                    input_t = F.linear(
                        input[i - 1].float(), self.weight if weights is None else weights
                    )
                if self.dropout_p:
                    input_t = F.dropout(input_t, p=self.dropout_p)

                I[i] = self.alpha * I[i - 1] + input_t
                V[i] = self.beta * V[i - 1] + (1 - self.beta) * I[i]

                V_spikes[i] = V[i]

                spikes = (V[i] > 1.0).float()
                V[i] = (1 - spikes) * V[i]

                # if self.reset_to_zero:
                #     V[i] = (1 - spikes) * V[i]
                # else:
                #     V[i] -= spikes

                output[i] = spikes
            is_silent = output.sum(0).mean(0) == 0
            if self.training and self.resolve_silent:
                self.weight.data[is_silent] += self.mu_silent
                if is_silent.sum() == 0:
                    break
            elif is_silent.sum() == 0:
                # print("WARNING : SILENT NEURONS")
                break
            else:
                break

        output_dict = {
            "spikes": output,
            "V": V,
            "I": I,
        }
        self.fwd_dict = output_dict
        pack = (input, V, V_spikes, I, output)
        return output, pack, output_dict

    def manual_backward(self, grad_output, pack, weights=None):
        # print("Grad Output is : ", grad_output.shape, grad_output.abs().max(0))
        input, _, V, I, post_spikes = pack
        # input = torch.roll(input, 1, dims=0)
        # post_spikes = torch.roll(post_spikes, 1, dims=0)
        steps = input.shape[0]
        if self.max_delay > 0:
            lV = torch.zeros(steps+self.max_delay-1, input.shape[1], self.output_dim).to(self.device)
            lI = torch.zeros(steps+self.max_delay-1, input.shape[1], self.output_dim).to(self.device)
        else:
            lV = torch.zeros(steps, input.shape[1], self.output_dim).to(self.device)
            lI = torch.zeros(steps, input.shape[1], self.output_dim).to(self.device)
        grad_input = torch.zeros(steps, input.shape[1], input.shape[2]).to(self.device)
        grad_weight = torch.zeros(input.shape[1], *self.weight.shape).to(self.device)
        jumps, V_dots = [], []

        if self.max_delay > 0:
            X = self.IDX - self.P 
            X = ((1 - X.abs()).relu()) #will use this for the actual gradient updates
            K = (X * self.weight)
            K = K.permute(2, 1, 0)
            

        for i in range(steps - 2, -1, -1):
            if self.max_delay > 0:
                grad_input[i] = F.conv1d((lV[i + 1:i+self.max_delay+1] - lI[i + 1:i+self.max_delay+1]).permute(1,2,0), K).permute(2,0,1)
            else:
                delta = lV[i + 1] - lI[i + 1]
                grad_input[i] = F.linear(delta, self.weight.t() if weights is None else weights.t())

            # Euler
            lI[i] = self.alpha * lI[i + 1] + (1 - self.alpha) * lV[i + 1]
            lV[i] = self.beta * lV[i + 1]

            # Jump
            V_dot = I[i + 1] - V[i + 1] + 1e-10
            V_dots.append(V_dot)
            jump = post_spikes[i + 1] * ((lV[i + 1] + grad_output[i + 1]) / V_dot)
            jumps.append(jump)

            if jump.mean().data.item() != 0:
                to_print = {
                    "jump": jump.data,
                    "grad_output": grad_output[i + 1].data,
                    "grad_input": grad_input[i].data,
                    "V_dot": (I[i] - V[i]).data,
                    "lV[i+1]": lV[i + 1].data,
                    "lI[i+1]": lI[i + 1].data,
                    "lV[i]": lV[i].data,
                    "lI[i]": lI[i].data,
                    # "I[i]": I[i].data,
                    # "I[i-1]": I[i - 1].data,
                    # "I[i+1]": I[i + 1].data,
                    # "V[i]": V[i].data,
                    # "V[i-1]": V[i - 1].data,
                    # "V[i+1]": V[i + 1].data,
                }

                # if self.output_dim == 3:
                # print(
                #     str(
                #         f"Got spike at time {i}, jump is {jump.cpu().data.numpy()}"
                #         + f"error is {grad_output[i + 1].cpu().data.numpy() }"
                #         + f"V_dot is {V_dot.cpu().data.numpy()}"
                #         + f"lV[i+1] is {lV[i + 1].cpu().data.numpy()}"
                #     )
                # )
                # print(to_print)

            lV[i] += jump

            # Accumulate grad
            spike_bool = input[i].float()
            if self.max_delay > 0:
                lI_delayed = (X.unsqueeze(1) * lI[i:i+self.max_delay].unsqueeze(-1)).sum(0)
                grad_weight -= self.tau_s * spike_bool.unsqueeze(1) * lI_delayed
            else:
                grad_weight -= self.tau_s * spike_bool.unsqueeze(1) * lI[i].unsqueeze(2) 
            # grad_weight -= spike_bool.unsqueeze(1)

        output_dict = {
            "lV": lV,
            "lI": lI,
            "jumps": jumps,
            "V_dots": V_dots,
            "grad_input": grad_input,
            "grad_weight": grad_weight,
        }

        # Store the backward dictionary with a limit on number kept
        if len(self.bwd_dicts) >= self.max_bwd_dicts:
            # Remove oldest item if we've reached the limit
            self.bwd_dicts.pop(0)

        self.bwd_dicts.append(output_dict)

        return grad_input.data, grad_weight.data, lV, lI

    def get_latest_bwd_dict(self):
        """
        Get the most recent backward dictionary.

        Returns:
            Dict or None: The most recent backward dictionary or None if none exists.
        """
        if self.bwd_dicts:
            return self.bwd_dicts[-1]
        return None

    def clear_bwd_dicts(self):
        """
        Clear all stored backward dictionaries.
        """
        self.bwd_dicts = []


class SpikingLinear_su(nn.Module):
    def __init__(self, d1, d2, **kwargs) -> None:
        super().__init__()
        self.input_dim, self.output_dim = d1, d2
        print(f"Creating Spiking Linear with {d1} -> {d2}")
        self.mu = kwargs.get("mu", 1.0)
        self.mu_silent = 1 / np.sqrt(d1) if d1 is not None else None
        self.resolve_silent = kwargs.get("resolve_silent", False)
        self.input_dropout_p = kwargs.get("input_dropout", None)
        if self.input_dropout_p:
            self.input_dropout = nn.Dropout(p=self.input_dropout_p)

        if d1 is not None:
            self.weight = nn.Parameter(torch.Tensor(d2, d1))
            nn.init.kaiming_normal_(self.weight)
            self.weight.data *= self.mu

        self.syn = snn.Synaptic(
            alpha=np.exp(-kwargs["dt"] / kwargs["tau_s"]),
            beta=np.exp(-kwargs["dt"] / kwargs["tau_m"]),
            init_hidden=True,
            reset_mechanism="zero",
            threshold=1.0,
        )

    def forward(self, input):
        if self.input_dim is None:
            self.input_dim = input.shape[-1]
            self.weight = nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
            nn.init.kaiming_normal_(self.weight)
            self.weight.data *= self.mu

        device = input.device

        if self.input_dropout_p:
            input = self.input_dropout(input)
        while True:
            steps = input.shape[0]
            V = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
            I = torch.zeros(steps, input.shape[1], self.output_dim).to(device)
            output = torch.zeros(steps, input.shape[1], self.output_dim).to(device)

            # out_spikes = []
            # voltages = []
            # currents = []

            for i in range(1, steps):
                out = F.linear(input[i - 1], self.weight)
                # for t, in_spikes in enumerate(input):
                # out = F.linear(in_spikes, self.weight)
                spikes = self.syn(out)

                output[i] = spikes
                V[i] = self.syn.mem
                I[i] = self.syn.syn

            if self.training and self.resolve_silent:
                is_silent = output.sum(0).mean(0) == 0
                self.weight.data[is_silent] += self.mu_silent
                if is_silent.sum() == 0:
                    break
            else:
                break

        out_dict = {"spikes": output, "V": V, "I": I}
        return output, out_dict

    def __repr__(self):
        return f"Spiking Linear ({self.input_dim}, {self.output_dim})"


layer_types = {str(l): l for l in [SpikingLinear_ev, SpikingLinear_su]}
model_types = {
    m: l for m, l in zip(["eventprop", "snntorch"], [SpikingLinear_ev, SpikingLinear_su])
}


class SpikeCELoss(nn.Module):
    def __init__(self, xi=1, alpha=0.0, beta=6.4, **kwargs):
        super().__init__()
        self.spike_time_fn = FirstSpikeTime.apply
        # self.spike_time_fn = SpikeTime().first_spike_fn
        self.xi = xi
        self.alpha = alpha
        self.beta = beta
        self.celoss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, input, target):
        if len(target.shape) == 0:
            target = target.unsqueeze(0)
        first_spikes = self.spike_time_fn(input)
        loss = self.celoss(-first_spikes / (self.xi), target)

        if self.alpha != 0:
            target_first_spike_times = first_spikes.gather(1, target.view(-1, 1))
            reg_loss = (
                loss + self.alpha * (torch.exp(target_first_spike_times / (self.beta)) - 1).mean()
            )
        else:
            reg_loss = loss

        return reg_loss, loss, first_spikes

    def __repr__(self):
        return f"SpikeCELoss(xi={self.xi}, alpha={self.alpha}, beta={self.beta})"


def softargmin1d(input, beta=10):
    *_, n = input.shape
    input = nn.functional.softmin(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


class SpikeQuadLoss(nn.Module):
    def __init__(self, xi=1, alpha=0.0, beta=6.4):
        super().__init__()
        self.spike_time_fn = FirstSpikeTime.apply
        self.xi = xi
        self.alpha = alpha
        self.beta = beta
        self.quadloss = nn.MSELoss(reduction="mean")

    def forward(self, input, target):
        if len(target.shape) == 0:
            target = target.unsqueeze(0)
        first_spikes = self.spike_time_fn(input)
        # loss = self.celoss(-first_spikes / (self.xi), target)
        soft_arg_spikes = softargmin1d(first_spikes)
        loss = self.quadloss(soft_arg_spikes, target.float())

        if self.alpha != 0:
            target_first_spike_times = first_spikes.gather(1, target.view(-1, 1))
            reg_loss = (
                loss + self.alpha * (torch.exp(target_first_spike_times / (self.beta)) - 1).mean()
            )
        else:
            reg_loss = loss

        return reg_loss, loss, first_spikes


class FirstSpikeTime(Function):
    @staticmethod
    def forward(input):
        idx = (
            torch.arange(input.shape[0], 0, -1).unsqueeze(-1).unsqueeze(-1).float().to(input.device)
        )
        first_spike_times = torch.argmax(idx * input, dim=0).float()
        assert not (first_spike_times == input.shape[0]).any(), "Last ts is not a spike time"
        first_spike_times[first_spike_times == 0] = input.shape[0]
        return first_spike_times

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        ctx.save_for_backward(*inputs, output.clone())

    @staticmethod
    def backward(ctx, grad_output):
        input, first_spike_times = ctx.saved_tensors
        k = (
            F.one_hot(first_spike_times.long().clip_(0, input.shape[0] - 1), input.shape[0])
            .float()
            .permute(2, 0, 1)
        )  # T x B x N
        k[-1] = 0.0  # Last ts is not a spike time
        # print(grad_output.shape, grad_output)
        grad_input = k * grad_output.unsqueeze(0)
        return grad_input


class SNN(nn.Module):
    def __init__(self, dims, **all_kwargs):
        super().__init__()

        self.get_first_spikes = all_kwargs.get("get_first_spikes", False)

        self.layer_type = all_kwargs.get("layer_type", None)
        self.model_type = all_kwargs.get("model_type", None)

        self.free_recordings = all_kwargs.get("free_recordings", True)

        assert not (self.layer_type is None and self.model_type is None), (
            "Must specify layer_type or model_type"
        )

        if self.model_type is not None:
            assert self.model_type in model_types, f"Invalid model_type {self.model_type}"
            layer = model_types[self.model_type]
            self.eventprop = self.model_type == "eventprop"
        else:
            assert self.layer_type in layer_types, f"Invalid layer_type {self.layer_type}"
            layer = layer_types[self.layer_type]
            self.eventprop = self.layer_type == str(SpikingLinear_ev)

        if all_kwargs.get("scale", None) is None:
            all_kwargs["scale"] = [
                [v_mu, v_sigma]
                for v_mu, v_sigma in zip(
                    [v for k, v in all_kwargs.items() if "mu" in k],
                    [v for k, v in all_kwargs.items() if "sigma" in k],
                )
            ]

        self.scale = all_kwargs.get("scale")

        layers = []
        if all_kwargs.get("seed", None) is not None:
            seed = all_kwargs.pop("seed")
            torch.manual_seed(seed)
            np.random.seed(seed)

        for i, (d1, d2) in enumerate(zip(dims[:-1], dims[1:])):
            layer_kwargs = {
                k: v[i] if isinstance(v, (list, np.ndarray)) else v for k, v in all_kwargs.items()
            }

            if i != 0:
                layer_kwargs["dropout"] = None

            layers.append(layer(d1, d2, **layer_kwargs))

        if self.get_first_spikes:
            self.outact = SpikeTime().first_spike_fn

        self.layers = RecordingSequential(*layers)

        # Only store the most recent backward dictionaries temporarily
        self._last_bwd_dicts = None

        # Debug flag to help with backward dictionary tracking
        self.debug = all_kwargs.get("debug", False)

    def to(self, device):
        self.device = device
        for l in self.layers:
            l.device = device
        return super().to(device)

    def forward(self, input, params=None, reset_recordings=True):
        """
        Forward pass through the SNN model.

        Args:
            input: Input tensor
            params: Optional parameters for meta-learning
            reset_recordings: Whether to reset recordings before forward pass

        Returns:
            Tuple of (output, layer_recordings)
        """
        if reset_recordings:
            self.reset_recordings()

        if not self.eventprop:
            input = input.float()
            reset(self)

        if len(input.shape) > 3:
            input = input.transpose(0, 1).squeeze()

        # Process forward pass
        out_dict = self.layers(input, params=params)
        out = out_dict["output"]

        # Explicitly gather backward dictionaries after forward pass
        # This is important for visualization later
        bwd_dicts = self.get_backward_dicts()

        # Debug prints to help diagnose backward dictionary issues
        if self.debug and len(bwd_dicts) > 0:
            print(f"Forward pass captured {len(bwd_dicts)} layer backward dictionaries")
            for layer_idx, layer_dicts in bwd_dicts.items():
                if isinstance(layer_dicts, list):
                    print(f"  Layer {layer_idx}: {len(layer_dicts)} backward dictionaries")
                else:
                    print(f"  Layer {layer_idx}: 1 backward dictionary")

        # Store for later retrieval (this will be moved to SNN after forward)
        self._last_bwd_dicts = bwd_dicts

        if self.get_first_spikes:
            out = self.outact(out)

        return out, out_dict["recordings"]

    def reset_recordings(self):
        """Reset backward dictionaries without accumulating them in memory."""
        # Reset backward dictionaries in each layer without storing them
        for layer in self.layers:
            if hasattr(layer, "bwd_dicts"):
                layer.bwd_dicts = []

        # Clear any stored backward dictionaries
        self._last_bwd_dicts = None

    def get_backward_dicts(self):
        """
        Get backward dictionaries from all layers.

        Returns:
            Dict: Dictionary of backward dictionaries organized by layer index.
        """
        bwd_dicts = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "bwd_dicts") and layer.bwd_dicts:
                bwd_dicts[i] = layer.bwd_dicts
        return bwd_dicts

    def get_layer_backward_dicts(self, layer_idx):
        """
        Get backward dictionaries for a specific layer.

        Args:
            layer_idx: Index of the layer to get backward dictionaries for.

        Returns:
            List: List of backward dictionaries for the specified layer.
        """
        if 0 <= layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            if hasattr(layer, "bwd_dicts"):
                return layer.bwd_dicts
        return []

    def get_last_backward_dicts(self):
        """
        Get the backward dictionaries from the last forward pass.

        Returns:
            Dict: Dictionary of backward dictionaries from the last forward pass.
        """
        # First try to get from stored attribute
        if self._last_bwd_dicts is not None:
            return self._last_bwd_dicts

        # If not available, try to collect from layers
        return self.get_backward_dicts()

    def meta_named_parameters(self, prefix="", recurse=True):
        gen = self._named_members(
            lambda module: (module._parameters.items() if isinstance(module, MetaModule) else []),
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            if elem[1].requires_grad:
                yield elem
