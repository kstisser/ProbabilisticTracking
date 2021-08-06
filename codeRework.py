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

    def get_default_args(self):
        self.num_frames = 5
        self.max_num_objects = 4
        self.expected_num_objects = 3.
        self.expected_num_spurious = 1.
        self.emission_prob = 0.8
        self.emission_noise_scale = 0.1 
        assert self.max_num_objects >= self.expected_num_objects
        return self 

    def get_prior_predictive_checks_args(self):
        self.num_frames = 5
        self.expected_num_objects = int(dist.Uniform(1, 5).sample())
        self.expected_num_spurious = float(int(dist.Uniform(1,3).sample()))
        self.max_num_objects = int(self.expected_num_objects + self.expected_num_spurious)
        self.emission_prob = max(0,min(1,abs((dist.Normal(0., 1.).rsample() + 3.)/4.))) #aiming for between [0.5,1.0]
        self.emission_noise_scale = max(0,min(1,abs((dist.Normal(0., 1.).rsample() + 2.)/4.))) #aiming for between [0.0,0.5]
        assert self.max_num_objects >= self.expected_num_objects
        print("Predictive check args. Num objects: ", self.expected_num_objects, " Num spurious: ", self.expected_num_spurious)
        print("Emission prob: ", self.emission_prob, " Noise scale: ", self.emission_noise_scale)
        return self


class DataGenerator:
    def __init__(self, movementType):
        print("Initializing DataGenerator with movement type ", movementType)
        self.movementType = movementType
        if self.movementType == MovementType.Sinusoidal:
            self.numDimensions = 2
        elif self.movementType == MovementType.Linear2D:
            self.numDimensions = 2
        elif self.movementType == MovementType.Linear3D:
            self.numDimensions = 3
        else:
            print("Error! Don't recognize movement type! ", self.movementType)

    def get_dynamics(self, num_frames):
        if self.movementType == MovementType.Sinusoidal:
            time = torch.arange(float(num_frames)) / 4
            dynamics = torch.stack([time.cos(), time.sin()], -1)
        elif self.movementType == MovementType.Linear2D:
            x = torch.arange(float(num_frames)) / 4
            y = torch.arange(float(num_frames)) / 4
            dynamics = torch.stack([x, y], -1)
        elif self.movementType == MovementType.Linear3D:
            x = torch.arange(float(num_frames)) / 4
            y = torch.arange(float(num_frames)) / 4
            z = torch.arange(float(num_frames)) / 4
            dynamics = torch.stack([x, y, z], -1)
        else:
            print("Error! Don't recognize movement type! ", self.movementType)

        return dynamics

    def generate_data(self, args):
        # Object model.
        num_objects = int(round(args.expected_num_objects))  
        print("Num frames: ", args.num_frames, " with num objects: ", num_objects)
        states = dist.Normal(0., 1.).sample((num_objects, self.numDimensions))

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
                states = pyro.sample("states", dist.Normal(0., 1.).expand([self.args.num_dimensions]).to_event(1))
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

                bernoulliRealProbs = self.args.expected_num_objects / num_observed
                bernoulliRealProbs = np.clip(bernoulliRealProbs, 0., 1.)
                pyro.sample("is_real",
                            dist.Bernoulli(bernoulliRealProbs),
                            obs=is_real.float())

                bernoulliSpuriousProbs = self.args.expected_num_spurious / num_observed
                bernoulliSpuriousProbs = np.clip(bernoulliSpuriousProbs, 0., 1.)
                pyro.sample("is_spurious",
                            dist.Bernoulli(bernoulliSpuriousProbs),
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
        states_loc = pyro.param("states_loc", lambda: torch.randn(int(self.args.max_num_objects), int(self.args.num_dimensions)))
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

    def train(self, true_states, true_positions, observations, visualize = False):
        pyro.set_rng_seed(0)
        true_num_objects = len(true_states)
        max_num_detections = observations.shape[1]
        print("generated {:d} detections from {:d} objects".format(
            (observations[..., -1] > 0).long().sum(), true_num_objects))

        pyro.set_rng_seed(1)
        pyro.clear_param_store()
        self.plot_solution( observations, true_positions, '(before training)')

        infer = SVI(self.model, self.guide, Adam({"lr": 0.01}), TraceEnum_ELBO(max_plate_nesting=2))
        losses = []
        for epoch in range(101):
            loss = infer.step(self.args, observations)
            if epoch % 10 == 0:
                print("epoch {: >4d} loss = {}".format(epoch, loss))
            if visualize:
                losses.append(loss)
        if visualize:
            pyplot.plot(losses)
            self.plot_solution(observations, true_positions, '(after training)')

    def get_predicted_positions(self):
        states_loc = pyro.param("states_loc")
        return self.dynamics.mm(states_loc.t())

    def plot_solution(self, observations, true_positions, message=''):
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


class PredictiveChecks:
    def __init__(self):
        self.argGenerator = Args()
        self.results = []
        self.dataGenerator = DataGenerator(MovementType.Linear2D)
        print("Initialized Predictive checks")

    def runPriorPredictiveChecks(self):
        self.results = []
        for i in range(500):
            args = self.argGenerator.get_prior_predictive_checks_args()
            print("Generated args for predictive checks")
            dynamics = self.dataGenerator.get_dynamics(args.num_frames)
            print("Generated dynamics for predictive checks")
            true_states, true_positions, observations, num_dimensions = self.dataGenerator.generate_data(args)
            args.num_dimensions = num_dimensions
            print("Generated data for predictive checks")

            targetProbabilisticModel = TargetProbabilisticModel(args, dynamics)
            print("Made model object")
            targetProbabilisticModel.train(true_states, true_positions, observations)
            print("Trained model")
            positions = targetProbabilisticModel.get_predicted_positions()
            print("Shape of positions: ", positions.shape())

            self.results.append((args, positions, true_positions))

    def plotResults(self):
        print("Would be plotting")

        #Plot when number of predictions doesn't match up with reality in number of objects
        numObject = []
        numPredictedObjects = []
        numObjectDiff = []
        emissionProbability = []
        numSpurious = []

        for args, positions, true_positions in self.results:
            numObject.append(args.num_objects)
            print("Positions shape: ", positions.shape)
            numPredictedObjects.append(positions.shape[1])
            numObjectDiff.append(numPredictedObjects[-1] - numObject[-1])
            emissionProbability.append(args.emission_prob)
            numSpurious.append(args.num_spurious)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(numObjectDiff, numSpurious, emissionProbability)

class PipelineManager:
    def __init__(self):
        self.argManager = Args()
        self.dataGenerator = DataGenerator(MovementType.Linear3D)        

    def trainAndPredict(self):
        args = self.argManager.get_default_args()
        dynamics = self.dataGenerator.get_dynamics(args.num_frames)

        true_states, true_positions, observations, num_dimensions = self.dataGenerator.generate_data(args)
        args.num_dimensions = int(num_dimensions)

        targetProbabilisticModel = TargetProbabilisticModel(args, dynamics)
        targetProbabilisticModel.train(true_states, true_positions, observations, True)
        targetProbabilisticModel.plot_solution(observations, true_positions)

    def runPriorPredictiveChecks(self):
        predictiveChecks = PredictiveChecks()
        predictiveChecks.runPriorPredictiveChecks()
        predictiveChecks.plotResults()

if __name__ == "__main__":
    pipelineManager = PipelineManager()
    #pipelineManager.runPriorPredictiveChecks()
    pipelineManager.trainAndPredict()


