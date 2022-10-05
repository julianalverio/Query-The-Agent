import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# from gpytorch.distributions import MultivariateNormal

# import sys
# import os
# import pathlib
# current = pathlib.Path(__file__).parent.resolve()
# sys.path.insert(0, os.path.join(current, '..'))
# from due.dkl import GP

from sklearn import cluster
import numpy as np

"""
the input x in both networks should be [observation, goal].
"""

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.action_max
        self.fc1 = nn.Linear(args.obs_dim + args.goal_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, args.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.action_max
        self.fc1 = nn.Linear(args.obs_dim + args.goal_dim + args.action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class MultipleOptimizer(object):
    def __init__(self, args, critics, forward=False, reward=False):
        self.args = args
        self.critics = critics
        self.optimizers = list()
        if forward:
            learning_rate = self.args.forward_lr
        elif reward:
            learning_rate = self.args.reward_lr
        else:
            learning_rate = self.args.lr_critic
        for critic in critics:
            self.optimizers.append(torch.optim.Adam(critic.parameters(), lr=learning_rate, weight_decay=self.args.critic_l2_norm))
        
    def zero_grad(self):
        for critic in self.critics:
            critic.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

class OLDGaussianCritic(nn.Module):
    def __init__(self, args):
        super(GaussianCritic, self).__init__()
        self.max_action = args.action_max
        self.fc1 = nn.Linear(args.obs_dim + args.goal_dim + args.action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 2)
        self.forward_out = nn.Linear(256, 2)
        self.reward_out = nn.Linear(256, 2)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.q_out(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6
        return mean, variance
    
    def predict_forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.forward_out(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6
        return mean, variance

    def predict_reward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.reward_out(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6
        return mean, variance
    
class OLDGaussianEnsembleCritic(nn.Module):
    def __init__(self, args):
        self.args = args
        super(GaussianEnsembleCritic, self).__init__()
        self.critics = [GaussianCritic(self.args) for _ in range(self.args.n_internal_critics)]
        self.optimizer = MultipleOptimizer(self.args, self.critics)
        self.forward_optimizer = MultipleOptimizer(self.args, self.critics, forward=True)
        self.reward_optimizer = MultipleOptimizer(self.args, self.critics, reward=True)

    def to(self, device):
        for critic in self.critics:
            critic.to(device)
        return self

    def copy_weights(self, source_network):
        for source_critic, local_critic in zip(source_network.critics, self.critics):
            local_critic.load_state_dict(source_critic.state_dict())

    def combine_gaussians(self, means, variances):
        true_mean = means.mean(axis=0).squeeze()
        true_variance = (variances + means.pow(2)).mean(axis=0).squeeze() - true_mean.pow(2)
        return true_mean, true_variance

    def forward(self, x, actions, combine=False):
        means, variances = list(), list()
        for critic in self.critics:
            mean, variance = critic(x, actions)
            means.append(mean)
            variances.append(variance)
        means = torch.stack(means, axis=0)
        variances = torch.stack(variances, axis=0)
        if combine:
            return self.combine_gaussians(means, variances)
        return means, variances

    def predict_forward(self, x, actions, combine=False):
        means, variances = list(), list()
        for critic in self.critics:
            mean, variance = critic.predict_forward(x, actions)
            means.append(mean)
            variances.append(variance)
        means = torch.stack(means, axis=0)
        variances = torch.stack(variances, axis=0)
        if combine:
            return self.combine_gaussians(means, variances)
        if self.args.use_gaussian_f:
            return means, variances
        else:
            return means

    def predict_reward(self, x, actions, combine=False):
        means, variances = list(), list()
        for critic in self.critics:
            mean, variance = critic.predict_reward(x, actions)
            means.append(mean)
            variances.append(variance)
        means = torch.stack(means, axis=0)
        variances = torch.stack(variances, axis=0)
        if combine:
            return self.combine_gaussians(means, variances)
        return means, variances

    def isolate_forward_head(self):
        for critic in self.critics:
            for name, param in self.critic_network.named_parameters():
                if 'forward' not in name:
                    param.requires_grad = False

    def isolate_reward_head(self):
        for critic in self.critics:
            for name, param in self.critic_network.named_parameters():
                if 'reward' not in name:
                    param.requires_grad = False

    def unfreeze_network(self):
        for critic in self.critics:
            for name, param in self.critic_network.named_parameters():
                param.requires_grad = True

# BOOTSTRAPPED
class MultiheadCritic(nn.Module):
    def __init__(self, args):
        super(MultiheadCritic, self).__init__()
        self.args = args
        self.num_heads = self.args.n_internal_critics
        self.max_action = args.action_max
        self.fc1 = nn.Linear(args.obs_dim + args.goal_dim + args.action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        for head_idx in range(self.num_heads):  # critic head
            setattr(self, f'q_out{head_idx}', nn.Linear(256, 1))  # critic head
            setattr(self, f'q_forward{head_idx}', nn.Linear(256, 1))  # forward head
            setattr(self, f'q_reward{head_idx}', nn.Linear(256, 1))  # reward head

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        outputs = list()
        for head_idx in range(self.num_heads):
            layer = getattr(self, f'q_out{head_idx}')
            outputs.append(layer(x))
        outputs = torch.stack(outputs)
        return outputs

    # same as forward but with a different head
    def predict_forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        state_preds = list()
        for critic_idx in range(self.args.n_internal_critics):
            state_preds.append(getattr(self, f'q_forward{critic_idx}')(x))
        return torch.stack(state_preds, axis=0)

    # same as forward but with a different head
    def predict_reward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        state_preds = list()
        for critic_idx in range(self.args.n_internal_critics):
            state_preds.append(getattr(self, f'q_reward{critic_idx}')(x))
        return torch.stack(state_preds, axis=0)


# Deep Ensembles
class GaussianEnsembleCritic(nn.Module):
    def __init__(self, args):
        super(GaussianEnsembleCritic, self).__init__()
        self.max_action = args.action_max
        self.n_critics = args.n_internal_critics
        self.optimizers = list()
        hidden_size = args.ensemble_hidden_size
        self.args = args

        for critic_idx in range(self.n_critics):
            setattr(self, f'q{critic_idx}_fc1', nn.Linear(args.obs_dim + args.goal_dim + args.action_dim, hidden_size))
            setattr(self, f'q{critic_idx}_fc2', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'q{critic_idx}_fc3', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'q{critic_idx}_mean', nn.Linear(hidden_size, 1))  # Q mean
            setattr(self, f'q{critic_idx}_variance', nn.Linear(hidden_size, 1))  # Q variance
            setattr(self, f'q{critic_idx}_forward', nn.Linear(hidden_size, args.goal_dim))  # forward mean
            setattr(self, f'q{critic_idx}_forward_variance', nn.Linear(hidden_size, 1))  # forward variance
            setattr(self, f'q{critic_idx}_reward', nn.Linear(hidden_size, 1))  # reward mean
            setattr(self, f'q{critic_idx}_reward_variance', nn.Linear(hidden_size, 1))  # reward variance
            
    # get Q values
    def forward(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        q_means, q_variances = list(), list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            q_out = getattr(self, f'q{critic_idx}_mean')
            q_value = q_out(x)
            q_variance_head = getattr(self, f'q{critic_idx}_variance')
            q_variance = q_variance_head(x)
            q_variance = F.softplus(q_variance) + 1e-6
            q_means.append(q_value)
            q_variances.append(q_variance)
        q_means = torch.stack(q_means)
        q_variances = torch.stack(q_variances)
        # combined_mean, combined_variance = self.combine_gaussians(q_means, q_variances)
        if self.args.use_gaussian_q:
            return q_means, q_variances
        else:
            return q_means

    def predict_forward(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        state_preds, forward_variances = list(), list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            forward_pred = getattr(self, f'q{critic_idx}_forward')(x)
            forward_variance = getattr(self, f'q{critic_idx}_forward_variance')(x)
            forward_variance = F.softplus(forward_variance) + 1e-6
            

            state_preds.append(forward_pred)
            forward_variances.append(forward_variance)
        state_preds = torch.stack(state_preds)
        forward_variances = torch.stack(forward_variances)
        if self.args.use_gaussian_f:
            return state_preds, forward_variances
        else:
            return state_preds
        # combined_means, combined_variances = self.combine_gaussians(state_preds, forward_variances)
        # return state_preds
        # return combined_means, combined_variances

    def predict_reward(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        reward_preds, reward_variances = list(), list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            reward_preds.append(getattr(self, f'q{critic_idx}_reward')(x))

            # reward_variance = getattr(self, f'q{critic_idx}_reward_variance')(x)
            # reward_variance = F.softplus(reward_variance) + 1e-6
            # reward_variances.append(reward_variance)

        reward_preds = torch.stack(reward_preds)
        # reward_variances = torch.stack(reward_variances)
        # return reward_preds, reward_variances
        return reward_preds

    def Q_idxs(self, x, actions, idxs):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        q_means = list()
        for idx in idxs:
            x = F.relu(getattr(self, f'q{idx}_fc1')(state_actions))
            x = F.relu(getattr(self, f'q{idx}_fc2')(x))
            x = F.relu(getattr(self, f'q{idx}_fc3')(x))
            q_mean = getattr(self, f'q{idx}_mean')(x)
            q_means.append(q_mean)
            
        return torch.stack(q_means)


class EnsembleCritic(nn.Module):
    def __init__(self, args):
        super(EnsembleCritic, self).__init__()
        self.max_action = args.action_max
        self.n_critics = args.n_internal_critics
        self.optimizers = list()
        hidden_size = args.ensemble_hidden_size
        self.args = args

        for critic_idx in range(self.n_critics):
            setattr(self, f'q{critic_idx}_fc1', nn.Linear(args.obs_dim + args.goal_dim + args.action_dim, hidden_size))
            setattr(self, f'q{critic_idx}_fc2', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'q{critic_idx}_fc3', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'q{critic_idx}_out', nn.Linear(hidden_size, 1))

            # reward head
            setattr(self, f'q{critic_idx}_reward', nn.Linear(hidden_size, 1))

            # forward head
            setattr(self, f'q{critic_idx}_forward', nn.Linear(hidden_size, args.goal_dim))

    def forward(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        q_values = list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            q_out = getattr(self, f'q{critic_idx}_out')
            q_value = q_out(x)
            q_values.append(q_value)
        return torch.stack(q_values, axis=0)

    def predict_forward(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        state_preds = list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            x = getattr(self, f'q{critic_idx}_forward')(x)

            state_preds.append(x)
        return torch.stack(state_preds, axis=0)

    def predict_reward(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        state_preds = list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            state_preds.append(getattr(self, f'q{critic_idx}_reward')(x))
        return torch.stack(state_preds, axis=0)

    def Q_idxs(self, x, actions, idxs):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        results = list()
        for idx in idxs:
            x = F.relu(getattr(self, f'q{idx}_fc1')(state_actions))
            x = F.relu(getattr(self, f'q{idx}_fc2')(x))
            x = F.relu(getattr(self, f'q{idx}_fc3')(x))
            x = getattr(self, f'q{idx}_out')(x)
            results.append(x)
        return torch.stack(results, axis=0)

    def Q1(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.q0_fc1(state_actions))
        x = F.relu(self.q0_fc2(x))
        x = F.relu(self.q0_fc3(x))
        return self.q0_out(x)


class GPEnsembleCritic(nn.Module):
    def __init__(self, args):
        super(GPEnsembleCritic, self).__init__()
        self.max_action = args.action_max
        self.n_critics = args.n_internal_critics
        hidden_size = args.ensemble_hidden_size
        self.args = args
        self.device = torch.device('cuda') if self.args.cuda else torch.device('cpu')

        for critic_idx in range(self.n_critics):
            setattr(self, f'q{critic_idx}_fc1', nn.Linear(args.obs_dim + args.goal_dim + args.action_dim, hidden_size))
            setattr(self, f'q{critic_idx}_fc2', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'q{critic_idx}_fc3', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'q{critic_idx}_out', nn.Linear(hidden_size, 1))

        self.gp_initialized = False

    def initialize_gp(self, inputs_tensor, actions_tensor):
        state_actions = torch.cat([inputs_tensor, actions_tensor / self.max_action], dim=1)
        with torch.no_grad():
            for critic_idx in range(self.n_critics):
                fc1 = getattr(self, f'q{critic_idx}_fc1')
                x = F.relu(fc1(state_actions))
                fc2 = getattr(self, f'q{critic_idx}_fc2')
                x = F.relu(fc2(x))
                fc3 = getattr(self, f'q{critic_idx}_fc3')
                x = F.relu(fc3(x))

                initial_lengthscale = torch.pdist(x).mean()

                kmeans = cluster.MiniBatchKMeans(
                    n_clusters=self.args.gp_inducing_points, batch_size=self.args.gp_inducing_points * 10
                )
                kmeans.fit(np.array(x.cpu()))
                initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_).cuda()
                gp = GP(num_outputs=1, initial_lengthscale=initial_lengthscale, initial_inducing_points=initial_inducing_points, kernel='RBF').cuda()
                setattr(self, f'q{critic_idx}_forward_gp_x', gp)
                gp = GP(num_outputs=1, initial_lengthscale=initial_lengthscale, initial_inducing_points=initial_inducing_points, kernel='RBF').cuda()
                setattr(self, f'q{critic_idx}_forward_gp_y', gp)

        self.gp_initialized = True
        
    def forward(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        q_values = list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            q_out = getattr(self, f'q{critic_idx}_out')
            q_value = q_out(x)
            q_values.append(q_value)
        return torch.stack(q_values, axis=0)

    def predict_forward(self, x, actions, distributions=False, evaluate=False):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        state_preds, state_variances = list(), list()
        x_distributions, y_distributions = list(), list()
        x_preds, x_variances, y_preds, y_variances = list(), list(), list(), list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            gp_x = getattr(self, f'q{critic_idx}_forward_gp_x')
            gp_y = getattr(self, f'q{critic_idx}_forward_gp_y')

            if evaluate:
                gp_x.eval()
                gp_y.eval()
            else:
                gp_x.train()
                gp_y.train()

            # transformation!
            # x = x[None, ...].repeat(2, 1, 1)

            x_distribution = gp_x(x, prior=False)
            x_pred = x_distribution.mean
            x_variance = x_distribution.variance
            y_distribution = gp_y(x, prior=False)
            y_pred = x_distribution.mean
            y_variance = y_distribution.variance
            
            x_distributions.append(x_distribution)
            x_preds.append(x_pred)
            x_variances.append(x_variance)
            y_distributions.append(y_distribution)
            y_preds.append(y_pred)
            y_variances.append(y_variance)
            
        if distributions:
            return torch.stack(x_preds), torch.stack(x_variances), x_distributions, torch.stack(y_preds), torch.stack(y_variances), y_distributions
        return torch.stack(x_preds), torch.stack(x_variances), torch.stack(y_preds), torch.stack(y_variances)

    def predict_reward(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        state_preds = list()
        for critic_idx in range(self.n_critics):
            fc1 = getattr(self, f'q{critic_idx}_fc1')
            x = F.relu(fc1(state_actions))
            fc2 = getattr(self, f'q{critic_idx}_fc2')
            x = F.relu(fc2(x))
            fc3 = getattr(self, f'q{critic_idx}_fc3')
            x = F.relu(fc3(x))
            state_preds.append(getattr(self, f'q{critic_idx}_reward')(x))
        return torch.stack(state_preds, axis=0)

    def Q_idxs(self, x, actions, idxs):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        results = list()
        for idx in idxs:
            x = F.relu(getattr(self, f'q{idx}_fc1')(state_actions))
            x = F.relu(getattr(self, f'q{idx}_fc2')(x))
            x = F.relu(getattr(self, f'q{idx}_fc3')(x))
            x = getattr(self, f'q{idx}_out')(x)
            results.append(x)
        return torch.stack(results, axis=0)

    def Q1(self, x, actions):
        state_actions = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.q0_fc1(state_actions))
        x = F.relu(self.q0_fc2(x))
        x = F.relu(self.q0_fc3(x))
        return self.q0_out(x)
