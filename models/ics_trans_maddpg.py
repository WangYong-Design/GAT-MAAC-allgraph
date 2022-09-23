import gc
import torch
from numpy.core.fromnumeric import repeat
import torch as th
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from memory_profiler import profile
from utilities.util import select_action
from models.model import Model
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GAT.GAT_encode_layer import GAT as EncoderLayer


class ICSTRANSMADDPG(Model):
    def __init__(self,args,target_net = None):
        super(ICSTRANSMADDPG, self).__init__(args)

        self.upper_bound = args.upper_bound

        # for observation Graph NN encoder
        self.obs_bus_dim = args.obs_bus_dim
        self.q_index = -1
        self.v_index = 2

        self.bus2region = args.bus2region   # region2number for every bus,example "zone1" to 1
        self.region_num = np.max(self.bus2region) + 1
        self.agent_index_in_state = args.agent_index_in_state
        self.encoder = EncoderLayer(self.obs_bus_dim + int(self.region_num), args)

        # For Graph NN topology
        self.edg_index = args.state_adj

        self.construct_model()
        self.apply(self.init_weights)

        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()

        self.batchnorm = nn.BatchNorm1d(self.n_).to(self.device)

    def construct_policy_net(self):
        # Graph NN + mlp head
        if self.args.agent_type == "GAT":
            from GAT.GAT_policy_ex import GATAgent
            Agent = GATAgent
        else:
            NotImplementedError()

        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([Agent(self.args)])
        else:
            self.policy_dicts = nn.ModuleList([Agent(self.args) for _ in range(self.n_)])

    def construct_value_net(self):
        if self.args.critic_type =="transformer":
            if self.args.critic_encoder:
                from transformer.transformer_critic_ex import TransformerCritic
                input_shape = self.args.hid_size + self.act_dim
                self.value_dicts = nn.ModuleList([TransformerCritic(input_shape,self.args)])
            else:
                from transformer.transformer_critic_ex import TransformerCritic
                input_shape = self.obs_bus_dim + self.act_dim
                self.value_dicts = nn.ModuleList([TransformerCritic(input_shape,self.args)])
        elif self.args.critic_type == "mlp":
                from critics.mlp_critic import MLPCritic
                input_shape = self.n_ * (self.obs_bus_dim + self.act_dim)
                self.value_dicts = nn.ModuleList([MLPCritic(input_shape,1,self.args)])
        else:
             NotImplementedError()

    def construct_auxiliary_net(self):
        if self.args.auxiliary:
            input_shape = self.args.hid_size
            output_shape = 1
            from transformer.transformer_aux_head import TransformerCritic as MLPHead
            self.auxiliary_dicts = nn.ModuleList( [ MLPHead(input_shape, output_shape, self.args, self.args.use_date) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()
        self.construct_auxiliary_net()

    def update_target(self):
        for name,params in self.target_net.value_dicts.state_dict().items():
            update_param = (1 - self.args.target_lr) * params + self.args.target_lr * self.value_dicts.state_dict()[name]
            self.target_net.value_dicts.state_dict()[name].copy_(update_param)
        for name,params in self.target_net.policy_dicts.state_dict().items():
            update_param = (1 - self.args.target_lr) * params + self.args.target_lr * self.policy_dicts.state_dict()[name]
            self.target_net.policy_dicts.state_dict()[name].copy_(update_param)
        if self.args.mixer:
            for name,params in self.target_net.mixer.state_dict().items():
                update_param = (1 - self.args.target_lr) * params + self.args.target_lr * self.mixer.state_dict()[name]
                self.target_net.mixer.state_dict()[name].copy_(update_param)
        if self.args.auxiliary:
            for name,params in self.target_net.auxiliary_dicts.state_dict().items():
                update_param = (1 - self.args.target_lr) * params + self.args.target_lr * self.auxiliary_dicts.state_dict()[name]
                self.target_net.auxiliary_dicts.state_dict()[name].copy_(update_param)

    def encode(self,raw_obs):
        # raw_obs (B,N,O)
        batch_size = raw_obs.shape[0]
        zone_id = F.one_hot(th.from_numpy(self.bus2region)).to(self.device).float() # (self.bus_num,region_num)
        zone_id = zone_id[None,:,:].contiguous().repeat(batch_size,1,1)
        agent_index = torch.from_numpy(self.agent_index_in_state)[None,:].repeat(batch_size,1).contiguous().to(self.device)

        input = torch.cat((raw_obs,zone_id),dim = -1)
        edg_index = self.edg_index * batch_size

        data_list = []
        for i in range(input.shape[0]):
            data_list.append(Data(input[i], edge_index=edg_index[i]))
        batch = torch_geometric.data.Batch.from_data_list(data_list).to(self.device)

        emb_agent_glimpsed = self.encoder(batch.x, batch.edge_index, agent_index)

        del data_list
        del edg_index
        del batch
        gc.collect()
        return emb_agent_glimpsed

    def policy(self,raw_obs,last_hid=None):
        # (B,N,O)
        batch_size = raw_obs.size(0)
        if self.args.shared_params:
            enc_obs = self.encode(raw_obs)

            agent_policy = self.policy_dicts[0]
            means,log_stds,hiddens = agent_policy(enc_obs,last_hid)
            means = means.contiguous().view(batch_size, self.n_, -1)
            hiddens = hiddens.contiguous().view(batch_size, self.n_, -1)
            if self.args.gaussian_policy:
                log_stds = log_stds.contiguous().view(batch_size, self.n_, -1)
            else:
                stds = th.ones_like(means).to(self.device) * self.args.fixed_policy_std
                log_stds = th.log(stds)
        else:
            NotImplementedError()

        return means,log_stds,hiddens


    def value(self,raw_obs,act):
        # raw_obs (B,N,O)
        # act     (B,N,act)
        batch_size = raw_obs.shape[0]
        if self.args.critic_encoder:
            if self.args.value_grad:
                emb_agent_glimpsed = self.encode(raw_obs).view(batch_size,self.n_,self.args.hid_size).contiguous()  # (B,n_,hid_size)
            else:
                with th.no_grad():
                    emb_agent_glimpsed = self.encode(raw_obs).view(batch_size,self.n_,self.args.hid_size).contiguous()

            inputs = torch.cat((emb_agent_glimpsed,act.contiguous()),dim=-1)
        else:
            obs = raw_obs.contiguous()
            act = act.contiguous()
            inputs = torch.cat((obs,act),dim = -1)

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            # data_list = []
            # edg_index = self.edg_index * batch_size
            #
            # for i in range(raw_obs.shpae[0]):
            #     data_list.append(Data(inputs[i], edge_index=edg_index[i]))
            # batch = torch_geometric.data.Batch.from_data_list(data_list).to(self.device)
            #
            # values, costs = agent_value(batch.x, batch.edge_index)     # (B,1)
            #
            # del data_list
            # del batch
            # gc.collect()

            values, costs = agent_value(inputs)
            values = values.contiguous().unsqueeze(dim=-1).repeat(1, self.n_, 1).view(batch_size, self.n_, 1)
            if self.args.critic_type == "mlp":
                costs = th.zeros_like(values)
            costs = costs.contiguous().view(batch_size, self.n_, 1)
        else:
            NotImplementedError()

        return values,costs

    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state,
                                                                                                              last_hid=last_hid)
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration,
                                                info={'log_std': log_stds_})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state,
                                                                                                        last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens


    def get_loss(self,batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False,
                                                                     actions_avail=actions_avail, target=False,
                                                                     last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)

        compose = self.value(state,actions)
        values,costs = compose[0].contiguous().view(-1,self.n_),compose[1].contiguous().view(-1,self.n_)

        compose = self.value(next_state,next_actions.detach())
        values_next,costs_next = compose[0].contiguous().view(-1,self.n_),compose[1].contiguous().view(-1,self.n_)

        compose = self.value(state,actions_pol)
        values_pol,costs_pol = compose[0].contiguous().view(-1,self.n_),compose[1].contiguous().view(-1,self.n_)

        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * values_next.detach()
        cost_returns = cost + self.args.cost_gamma * (1 - done) * costs_next.detach()

        deltas,cost_deltas = returns - values,cost_returns - costs           # Bellman equation

        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)

        policy_loss = -advantages
        policy_loss = policy_loss.mean()
        critic_loss = deltas.pow(2).mean()
        # lambda_loss = - ((cost_returns.detach() - self.upper_bound) * self.multiplier).mean(dim=0).sum()

        return policy_loss, critic_loss, action_out, None

    def get_auxiliary_loss(self,batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)

        state_ = state.view(batch_size,self.bus_num,self.obs_bus_dim).contiguous()
        with torch.no_grad():
            labels = self._cal_out_of_control(state_)

        enc_obs = self.encode(state_)
        preds,_ = self.auxiliary_dicts[0](enc_obs,None)
        loss = nn.MSELoss(preds,labels)

        return loss

    def _cal_out_of_control(self,state):
        v = state[:,:,self.v_index]
        out_of_control = th.logical_or(v < 0.95, v > 1.05).float()
        percentage_out_of_controal = th.sum(out_of_control,dim=1,keepdim=True)/self.bus_num
        return percentage_out_of_controal


