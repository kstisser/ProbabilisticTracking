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

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

assert pyro.__version__.startswith('1.7.0')
smoke_test = ('CI' in os.environ)

def get_dynamics(num_frames):
    time = torch.arange(float(num_frames)) / 4
    return torch.stack([time.cos(), time.sin()], -1)

def generate_data(args):
    # Object model.
    num_objects = int(round(args.expected_num_objects))  # Deterministic.
    states = dist.Normal(0., 1.).sample((num_objects, 2))

    # Detection model.
    emitted = dist.Bernoulli(args.emission_prob).sample((args.num_frames, num_objects))
    num_spurious = dist.Poisson(args.expected_num_spurious).sample((args.num_frames,))
    max_num_detections = int((num_spurious + emitted.sum(-1)).max())
    observations = torch.zeros(args.num_frames, max_num_detections, 1+1) # position+confidence
    positions = get_dynamics(args.num_frames).mm(states.t())
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

    return states, positions, observations

def model(args, observations):
    with pyro.plate("objects", args.max_num_objects):
        exists = pyro.sample("exists",
                             dist.Bernoulli(args.expected_num_objects / args.max_num_objects))
        with poutine.mask(mask=exists.bool()):
            states = pyro.sample("states", dist.Normal(0., 1.).expand([2]).to_event(1))
            positions = get_dynamics(args.num_frames).mm(states.t())
    with pyro.plate("detections", observations.shape[1]):
        with pyro.plate("time", args.num_frames):
            # The combinatorial part of the log prob is approximated to allow independence.
            is_observed = (observations[..., -1] > 0)
            with poutine.mask(mask=is_observed):
                assign = pyro.sample("assign",
                                     dist.Categorical(torch.ones(args.max_num_objects + 1)))
            is_spurious = (assign == args.max_num_objects)
            is_real = is_observed & ~is_spurious
            num_observed = is_observed.float().sum(-1, True)
            pyro.sample("is_real",
                        dist.Bernoulli(args.expected_num_objects / num_observed),
                        obs=is_real.float())
            pyro.sample("is_spurious",
                        dist.Bernoulli(args.expected_num_spurious / num_observed),
                        obs=is_spurious.float())

            # The remaining continuous part is exact.
            observed_positions = observations[..., 0]
            with poutine.mask(mask=is_real):
                bogus_position = positions.new_zeros(args.num_frames, 1)
                augmented_positions = torch.cat([positions, bogus_position], -1)
                predicted_positions = gather(augmented_positions, assign, -1)
                pyro.sample("real_observations",
                            dist.Normal(predicted_positions, args.emission_noise_scale),
                            obs=observed_positions)
            with poutine.mask(mask=is_spurious):
                pyro.sample("spurious_observations", dist.Normal(0., 1.),
                            obs=observed_positions)

def guide(args, observations):
    # Initialize states randomly from the prior.
    states_loc = pyro.param("states_loc", lambda: torch.randn(args.max_num_objects, 2))
    states_scale = pyro.param("states_scale",
                              lambda: torch.ones(states_loc.shape) * args.emission_noise_scale,
                              constraint=constraints.positive)
    positions = get_dynamics(args.num_frames).mm(states_loc.t())

    # Solve soft assignment problem.
    real_dist = dist.Normal(positions.unsqueeze(-2), args.emission_noise_scale)
    spurious_dist = dist.Normal(0., 1.)
    is_observed = (observations[..., -1] > 0)
    observed_positions = observations[..., 0].unsqueeze(-1)
    assign_logits = (real_dist.log_prob(observed_positions) -
                     spurious_dist.log_prob(observed_positions) +
                     math.log(args.expected_num_objects * args.emission_prob /
                              args.expected_num_spurious))
    assign_logits[~is_observed] = -float('inf')
    exists_logits = torch.empty(args.max_num_objects).fill_(
        math.log(args.max_num_objects / args.expected_num_objects))
    assignment = MarginalAssignmentPersistent(exists_logits, assign_logits)

    with pyro.plate("objects", args.max_num_objects):
        exists = pyro.sample("exists", assignment.exists_dist, infer={"enumerate": "parallel"})
        with poutine.mask(mask=exists.bool()):
            pyro.sample("states", dist.Normal(states_loc, states_scale).to_event(1))
    with pyro.plate("detections", observations.shape[1]):
        with poutine.mask(mask=is_observed):
            with pyro.plate("time", args.num_frames):
                assign = pyro.sample("assign", assignment.assign_dist, infer={"enumerate": "parallel"})

    return assignment


def plot_solution(message=''):
    assignment = guide(args, observations)
    states_loc = pyro.param("states_loc")
    positions = get_dynamics(args.num_frames).mm(states_loc.t())
    pyplot.figure(figsize=(12,6)).patch.set_color('white')
    pyplot.plot(true_positions.numpy(), 'k--')
    is_observed = (observations[..., -1] > 0)
    pos = observations[..., 0]
    time = torch.arange(float(args.num_frames)).unsqueeze(-1).expand_as(pos)
    pyplot.scatter(time[is_observed].view(-1).numpy(),
                   pos[is_observed].view(-1).numpy(), color='k', marker='+',
                   label='observation')
    for i in range(args.max_num_objects):
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

args = type('Args', (object,), {})  # A fake ArgumentParser.parse_args() result.

args.num_frames = 5
args.max_num_objects = 3
args.expected_num_objects = 2.
args.expected_num_spurious = 1.
args.emission_prob = 0.8
args.emission_noise_scale = 0.1

assert args.max_num_objects >= args.expected_num_objects

pyro.set_rng_seed(0)
true_states, true_positions, observations = generate_data(args)
true_num_objects = len(true_states)
max_num_detections = observations.shape[1]
assert true_states.shape == (true_num_objects, 2)
assert true_positions.shape == (args.num_frames, true_num_objects)
assert observations.shape == (args.num_frames, max_num_detections, 1+1)
print("generated {:d} detections from {:d} objects".format(
    (observations[..., -1] > 0).long().sum(), true_num_objects))



pyro.set_rng_seed(1)
pyro.clear_param_store()
plot_solution('(before training)')

infer = SVI(model, guide, Adam({"lr": 0.01}), TraceEnum_ELBO(max_plate_nesting=2))
losses = []
for epoch in range(101 if not smoke_test else 2):
    loss = infer.step(args, observations)
    if epoch % 10 == 0:
        print("epoch {: >4d} loss = {}".format(epoch, loss))
    losses.append(loss)
pyplot.plot(losses)
plot_solution('(after training)')