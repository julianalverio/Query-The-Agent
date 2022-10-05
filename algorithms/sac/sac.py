import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.model import GaussianPolicy, QNetwork

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
class SAC(object):
    def __init__(self, state_dim, action_space, configs, device):
        self.gamma = configs.gamma
        self.tau = configs.tau
        self.alpha = configs.alpha
        self.reward_scale = configs.reward_scale

        self.automatic_entropy_tuning = configs.automatic_entropy_tuning

        self.device = device

        self.critic = QNetwork(state_dim, action_space.shape[0], configs.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=configs.lr)

        self.critic_target = QNetwork(state_dim, action_space.shape[0], configs.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=configs.lr)

        self.policy = GaussianPolicy(state_dim, action_space.shape[0], configs.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=configs.lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self.critic_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        # else:
            # alpha_loss = torch.tensor(0.).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        soft_update(self.critic_target, self.critic, self.tau)

    # Save model parameters    
    def save_model(self, actor_path, critic_path):
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        policy_state_dict = torch.load(actor_path)
        critic_state_dict = torch.load(critic_path)
        self.policy.load_state_dict(policy_state_dict)
        self.critic.load_state_dict(critic_state_dict)
        

