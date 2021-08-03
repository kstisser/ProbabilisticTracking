import math
import os
import torch
from torch.distributions import constraints
from matplotlib import pyplot

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.tracking.assignment import MarginalAssignmentPersistent
from pyro.distributions.util import gather
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
import numpy as np
import enum

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

assert pyro.__version__.startswith('1.7.0')

class MovementType(enum.Enum):
    Sinusoidal = 1
    Linear2D = 2
    Linear3D = 3

class Args:
    def __init__(self):
        self.args = self.get_default_args()

    def parseInputArgs(self, args):

        return self

    def get_default_args(self):
        self.num_frames = 5
        self.max_num_objects = 4
        self.expected_num_objects = 3.
        self.expected_num_spurious = 1.
        self.emission_prob = 0.8
        self.emission_noise_scale = 0.1 
        return self       

class DataGenerator:
    def __init__(self, movementType):
        print("Initializing DataGenerator with movement type ", movementType)
        self.movementType = movementType
        if self.movementType == MovementType.Sinusoidal:
            self.numDimensions = 2
        elif self.movementType == MovementType.Linear2D:
            self.numDimensions = 2
        elif self.movemetType == MovementType.Linear3D:
            self.numDimensions = 3
        else:
            print("Error! Don't recognize movement type! ", self.movementType)

    def get_dynamics(self, num_frames):
        if self.movementType == MovementType.Sinusoidal:
            time = torch.arange(float(num_frames)) / 4
            transposed = torch.stack([time.cos(), time.sin()], -1)
        elif self.movementType == MovementType.Linear2D:
            x = torch.arange(float(num_frames)) / 4
            y = torch.arange(float(num_frames)) / 4
            transposed = torch.stack([x, y], -1)
        elif self.movemetType == MovementType.Linear3D:
            x = torch.arange(float(num_frames)) / 4
            y = torch.arange(float(num_frames)) / 4
            z = torch.arange(float(num_frames)) / 4
            transposed = torch.stack([x, y, z], -1)
        else:
            print("Error! Don't recognize movement type! ", self.movementType)

        return transposed

    def generate_data(self, args):
        # Object model.
        num_objects = int(round(args.expected_num_objects))  # Deterministic.
        states = dist.Normal(0., 1.).sample((num_objects, self.numDimensions))

        print("Num frames: ", args.num_frames, " with num objects: ", num_objects)
        # Detection model.
        emitted = dist.Bernoulli(args.emission_prob).sample((args.num_frames, num_objects))
        num_spurious = dist.Poisson(args.expected_num_spurious).sample((args.num_frames,))
        max_num_detections = int((num_spurious + emitted.sum(-1)).max())
        observations = torch.zeros(args.num_frames, max_num_detections, 2) # position+confidence
        positions = self.get_dynamics(args.num_frames).mm(states.t())

        noisy_positions = dist.Normal(positions, args.emission_noise_scale).sample()
        for t in range(args.num_frames):
            j = 0
            for i, e in enumerate(emitted[t]):
                if e:
                    observations[t, j, 0] = noisy_positions[t, i]
                    observations[t, j, 1] = 1
                    j += 1
            n = int(num_spurious[t])
            if n:
                observations[t, j:j+n, 0] = dist.Normal(0., 1.).sample((n,))
                observations[t, j:j+n, 1] = 1

        return states, positions, observations, self.numDimensions

class TargetProbabilisticModel:
    def __init__(self, args, dynamics):
        self.args = args
        self.dynamics = dynamics

    def model(self, args, observations):
        with pyro.plate("objects", self.args.max_num_objects):
            exists = pyro.sample("exists",
                                dist.Bernoulli(self.args.expected_num_objects / self.args.max_num_objects))
            with poutine.mask(mask=exists.bool()):
                states = pyro.sample("states", dist.Normal(0., 1.).expand([2]).to_event(1))
                positions = self.dynamics.mm(states.t())
        with pyro.plate("detections", observations.shape[1]):
            with pyro.plate("time", self.args.num_frames):
                # The combinatorial part of the log prob is approximated to allow independence.
                is_observed = (observations[..., -1] > 0)
                with poutine.mask(mask=is_observed):
                    assign = pyro.sample("assign",
                                        dist.Categorical(torch.ones(self.args.max_num_objects + 1)))
                is_spurious = (assign == self.args.max_num_objects)
                is_real = is_observed & ~is_spurious
                num_observed = is_observed.float().sum(-1, True)
                pyro.sample("is_real",
                            dist.Bernoulli(self.args.expected_num_objects / num_observed),
                            obs=is_real.float())
                pyro.sample("is_spurious",
                            dist.Bernoulli(self.args.expected_num_spurious / num_observed),
                            obs=is_spurious.float())

                # The remaining continuous part is exact.
                observed_positions = observations[..., 0]
                with poutine.mask(mask=is_real):
                    bogus_position = positions.new_zeros(self.args.num_frames, 1)
                    augmented_positions = torch.cat([positions, bogus_position], -1)
                    predicted_positions = gather(augmented_positions, assign, -1)
                    pyro.sample("real_observations",
                                dist.Normal(predicted_positions, self.args.emission_noise_scale),
                                obs=observed_positions)
                with poutine.mask(mask=is_spurious):
                    pyro.sample("spurious_observations", dist.Normal(0., 1.),
                                obs=observed_positions)

    def guide(self, args, observations):
        # Initialize states randomly from the prior.
        states_loc = pyro.param("states_loc", lambda: torch.randn(self.args.max_num_objects, num_dimensions))
        states_scale = pyro.param("states_scale",
                                lambda: torch.ones(states_loc.shape) * self.args.emission_noise_scale,
                                constraint=constraints.positive)
        positions = self.dynamics.mm(states_loc.t())

        # Solve soft assignment problem.
        real_dist = dist.Normal(positions.unsqueeze(-2), self.args.emission_noise_scale)
        spurious_dist = dist.Normal(0., 1.)
        is_observed = (observations[..., -1] > 0)
        observed_positions = observations[..., 0].unsqueeze(-1)
        assign_logits = (real_dist.log_prob(observed_positions) -
                        spurious_dist.log_prob(observed_positions) +
                        math.log(self.args.expected_num_objects * self.args.emission_prob /
                                self.args.expected_num_spurious))
        assign_logits[~is_observed] = -float('inf')
        exists_logits = torch.empty(self.args.max_num_objects).fill_(
            math.log(self.args.max_num_objects / self.args.expected_num_objects))
        assignment = MarginalAssignmentPersistent(exists_logits, assign_logits)

        with pyro.plate("objects", self.args.max_num_objects):
            exists = pyro.sample("exists", assignment.exists_dist, infer={"enumerate": "parallel"})
            with poutine.mask(mask=exists.bool()):
                pyro.sample("states", dist.Normal(states_loc, states_scale).to_event(1))
        with pyro.plate("detections", observations.shape[1]):
            with poutine.mask(mask=is_observed):
                with pyro.plate("time", self.args.num_frames):
                    assign = pyro.sample("assign", assignment.assign_dist, infer={"enumerate": "parallel"})

        return assignment

    def train(self, true_states, true_positions, observations):
        pyro.set_rng_seed(0)
        true_num_objects = len(true_states)
        max_num_detections = observations.shape[1]
        print("generated {:d} detections from {:d} objects".format(
            (observations[..., -1] > 0).long().sum(), true_num_objects))

        pyro.set_rng_seed(1)
        pyro.clear_param_store()
        self.plot_solution('(before training)')

        infer = SVI(self.model, self.guide, Adam({"lr": 0.01}), TraceEnum_ELBO(max_plate_nesting=2))
        losses = []
        for epoch in range(101):
            loss = infer.step(self.args, observations)
            if epoch % 10 == 0:
                print("epoch {: >4d} loss = {}".format(epoch, loss))
            losses.append(loss)
        pyplot.plot(losses)
        self.plot_solution('(after training)')

    def plot_solution(self, message=''):
        assignment = self.guide(self.args, observations)
        states_loc = pyro.param("states_loc")
        positions = self.dynamics.mm(states_loc.t())
        pyplot.figure(figsize=(12,6)).patch.set_color('white')
        pyplot.plot(true_positions.numpy(), 'k--')
        is_observed = (observations[..., -1] > 0)
        pos = observations[..., 0]
        time = torch.arange(float(self.args.num_frames)).unsqueeze(-1).expand_as(pos)
        pyplot.scatter(time[is_observed].view(-1).numpy(),
                    pos[is_observed].view(-1).numpy(), color='k', marker='+',
                    label='observation')
        for i in range(self.args.max_num_objects):
            p_exist = assignment.exists_dist.probs[i].item()
            position = positions[:, i].detach().numpy()
            pyplot.plot(position, alpha=p_exist, color='C0')
        pyplot.title('Truth, observations, and predicted tracks ' + message)
        pyplot.plot([], 'k--', label='truth')
        pyplot.plot([], color='C0', label='prediction')
        pyplot.legend(loc='best')
        pyplot.xlabel('time step')
        pyplot.ylabel('position')
        pyplot.tight_layout()
        pyplot.show()

if __name__ == "__main__":
    argsObs = Args()
    args = argsObs.get_default_args()
    assert args.max_num_objects >= args.expected_num_objects

    dataGenerator = DataGenerator(MovementType.Linear2D)
    dynamics = dataGenerator.get_dynamics(args.num_frames)
    true_states, true_positions, observations, num_dimensions = dataGenerator.generate_data(args)
    args.num_dimensions = num_dimensions

    targetProbabilisticModel = TargetProbabilisticModel(args, dynamics)
    targetProbabilisticModel.train(true_states, true_positions, observations)
    targetProbabilisticModel.plot_solution()

