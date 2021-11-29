import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.fmac_critic import FMACCritic
from modules.critics.lica import LICACritic
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.alita import ALITAMixer
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_td_lambda_targets
from components.epsilon_schedules import DecayThenFlatSchedule
from utils.th_utils import get_parameters_num


class ALITALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = FMACCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())


        self.qclone = FMACCritic(scheme, args)
        self.target_qclone = copy.deepcopy(self.critic)
        self.qclone_params = list(self.critic.parameters())
        
        self.mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "alita":
                self.mixer = ALITAMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.critic_params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        print('Mixer Size: ')
        print(get_parameters_num(self.critic_params))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.qclone_optimiser = RMSprop(params=self.qclone_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
            self.qclone_optimiser = Adam(params=self.qclone_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0



    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, off=False):
        # Get the relevant data
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Retrace Q target
        with th.no_grad():
            q1, _ = self.target_critic(batch,  batch["actions_onehot"].detach())
            target_vals = self.target_mixer(q1, batch["state"])
            
            lambd = 0 if off else self.args.lambd
            target_vals = build_td_lambda_targets(rewards, 
                    terminated, mask, target_vals, self.n_agents, self.args.gamma, lambd)

        # Train the critic
        # Current Q network forward
        q1, _ = self.critic(batch[:, :-1], actions_onehot.detach())
        q_taken = self.mixer(q1, batch["state"][:,:-1])
        critic_loss = 0.5 * ((q_taken - target_vals.detach()) * mask).pow(2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()


        ################################# qclones here
        
        with th.no_grad():
            q1, _ = self.target_qclone(batch,  batch["actions_onehot"].detach())
            target_vals = self.target_mixer(q1, batch["state"])
            
            lambd = 0 if off else self.args.lambd
            target_vals = build_td_lambda_targets(rewards, 
                    terminated, mask, target_vals, self.n_agents, self.args.gamma, lambd) ############Change Here OR


        
        
        q1, _ = self.qclone(batch[:, :-1], actions_onehot.detach())

        
        targets = target_vals                                                           ###########Or Here

        # Td-error
        td_error = (q1 - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        qclone_loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()
        
        # Optimise
        self.qclone_optimiser.zero_grad()
        qclone_loss.backward()
        qclone_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.qclone_optimiser.step()
        ##############

        
        #########################################
        # Train the actor
        if not off:
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.mac.forward(batch, t=t)
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1)  # Concat over time b, t, a, probs

            q1, _ = self.critic(batch[:,:-1], pi)
            q = self.mixer(q1, batch["state"][:, :-1])
            pg_loss = -(q * mask).sum() / mask.sum() 

            entropy_loss = categorical_entropy(pi).mean(-1, keepdim=True) # mean over agents
            entropy_loss[mask == 0] = 0 # fill nan
            entropy_loss = (entropy_loss* mask).sum() / mask.sum()

            ####################
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_qclone.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999


                # Max over target Q-Values
            if self.args.double_q:
                mac_out=pi
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]


            influence=((q1 - target_max_qvals.detach())** 2).sum() / mask.sum()

            ####################
            loss = pg_loss - self.args.entropy_coef * entropy_loss / entropy_loss.item() +self.args.inf*influence

            self.agent_optimiser.zero_grad()
            loss.backward()
            agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        # target_update
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # log
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm.item(), t_env)
            self.logger.log_stat("target_vals", (target_vals * mask).sum().item() / mask.sum().item(), t_env)

            if not off:
                self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
                self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
                self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
                agent_mask = mask.repeat(1, 1, self.n_agents)
                self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * agent_mask).sum().item() / agent_mask.sum().item(), t_env)
                self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        # self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda"):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.to(device=device)
        self.target_critic.to(device=device)
        if self.mixer is not None:
            self.mixer.to(device=device)
            self.target_mixer.to(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))