import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Plotter:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self, save_path=None):
        self.plot_process = Process(target=self._plot, kwargs={"save_path": save_path})
        self.plot_process.start()

    def plot_additional_states(self):
        self.plot_pos_process = Process(target=self._plot_pos)
        self.plot_pos_process.start()
        self.plot_vel_process = Process(target=self._plot_vel)
        self.plot_vel_process.start()
        self.plot_torque_process = Process(target=self._plot_torque)
        self.plot_torque_process.start()

    def _plot(self, save_path=None):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(28, 14))
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log

        # plot
        a = axs[0, 0]
        for key, value in log.items():
            if key.startswith("reward_"):
                a.plot(
                    time,
                    value,
                    label=key,
                    alpha=0.5,
                )
        a.set(xlabel="time [s]", ylabel="Rewards", title="Termwise Rewards")
        a.legend()

        # plot
        a = axs[0, 1]
        if "joint_vel_max" in log:
            a.plot(
                time,
                log["joint_vel_max"],
                label="joint_vel",
            )
        a.set(xlabel="time [s]", ylabel="Max Joint Vel", title="Max Joint Velocities")
        a.legend()

        # plot
        a = axs[0, 2]
        if "joint_acc_max" in log:
            a.plot(
                time,
                log["joint_acc_max"],
                label="joint_acc",
            )
        a.set(xlabel="time [s]", ylabel="Max Joint Acc", title="Max Joint Accelerations")
        a.legend()

        # plot
        a = axs[1, 0]
        for key, value in log.items():
            if key.startswith("reward_"):
                a.plot(
                    time,
                    -np.sqrt(-np.log(value)),
                    label=key,
                    alpha=0.5,
                )
        a.set(xlabel="time [s]", ylabel="log Rewards", title="Termwise log Rewards")
        a.legend()

        # plot
        a = axs[1, 1]
        if "action_rate" in log:
            a.plot(
                time,
                log["action_rate"],
                label="action_rate",
                alpha=0.6,
            )
        if "action_rate_max" in log:
            a.plot(
                time,
                log["action_rate_max"],
                label="action_rate_max",
            )
        a.set(xlabel="time [s]", ylabel="", title="Overall Action Rate")
        a.legend()

        # # plot
        a = axs[1, 2]
        for key, value in log.items():
            if key.startswith("scaled_action_"):
                a.plot(
                    time,
                    value,
                    label=key,
                )
            if key.startswith("joint_position_"):
                a.plot(
                    time,
                    value,
                    label=key,
                )
            if key.startswith("joint_ref_"):
                a.plot(
                    time,
                    value,
                    label=key,
                )
        a.set(xlabel="time [s]", ylabel="position [rad]", title="Joint Action target and Joint Position")
        a.legend()

        # # plot
        a = axs[2, 0]
        if "roll" in log:
            a.plot(
                time,
                log["roll"],
                label="roll",
            )
        if "pitch" in log:
            a.plot(
                time,
                log["pitch"],
                label="pitch",
            )
        if "yaw" in log:
            a.plot(
                time,
                log["yaw"],
                label="yaw",
            )
        a.set(xlabel="time [s]", ylabel="roll/pitch [rad]", title="Robot Orientation")
        a.legend()

        # # plot
        a = axs[2, 1]
        if "proj_grav_x" in log:
            a.plot(
                time,
                log["proj_grav_x"],
                label="proj_grav_x",
            )
        if "proj_grav_y" in log:
            a.plot(
                time,
                log["proj_grav_y"],
                label="proj_grav_y",
            )
        if "proj_grav_z" in log:
            a.plot(
                time,
                log["proj_grav_z"],
                label="proj_grav_z",
            )
        a.set(xlabel="time [s]", ylabel="", title="Projected Gravity Vector")
        a.legend()

        # # plot
        a = axs[2, 2]
        for key, value in log.items():
            if key.startswith("joint_torque_"):
                a.plot(
                    time,
                    value,
                    label=key,
                )
        a.set(xlabel="time [s]", ylabel="tau [Nm]", title="Joint Applied Torques")
        a.legend()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def _plot_pos(self):
        log = self.state_log
        nb_rows = int(np.sqrt(log["all_dof_pos"][0].shape[0]))
        nb_cols = int(np.ceil(log["all_dof_pos"][0].shape[0] / nb_rows))
        nb_rows, nb_cols = nb_cols, nb_rows
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        # plot joint positions
        for i in range(nb_rows):
            for j in range(nb_cols):
                if i * nb_cols + j < log["all_dof_pos"][0].shape[0]:
                    a = axs[i][j]
                    a.plot(
                        time,
                        [all_dof_pos[i * nb_cols + j] for all_dof_pos in log["all_dof_pos"]],
                        label="measured",
                    )
                    a.set(xlabel="time [s]", ylabel="Position [rad]", title=f"Joint Position {i*nb_cols+j}")
                    a.legend()
                else:
                    break
        plt.show()

    def _plot_vel(self):
        log = self.state_log
        nb_rows = int(np.sqrt(log["all_dof_vel"][0].shape[0]))
        nb_cols = int(np.ceil(log["all_dof_vel"][0].shape[0] / nb_rows))
        nb_rows, nb_cols = nb_cols, nb_rows
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        # plot joint velocities
        for i in range(nb_rows):
            for j in range(nb_cols):
                if i * nb_cols + j < log["all_dof_vel"][0].shape[0]:
                    a = axs[i][j]
                    a.plot(
                        time,
                        [all_dof_vel[i * nb_cols + j] for all_dof_vel in log["all_dof_vel"]],
                        label="measured",
                    )
                    a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title=f"Joint Velocity {i*nb_cols+j}")
                    a.legend()
                else:
                    break
        plt.show()

    def _plot_torque(self):
        log = self.state_log
        nb_rows = int(np.sqrt(log["all_dof_torque"][0].shape[0]))
        nb_cols = int(np.ceil(log["all_dof_torque"][0].shape[0] / nb_rows))
        nb_rows, nb_cols = nb_cols, nb_rows
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        # plot joint torques
        for i in range(nb_rows):
            for j in range(nb_cols):
                if i * nb_cols + j < log["all_dof_torque"][0].shape[0]:
                    a = axs[i][j]
                    a.plot(
                        time,
                        [all_dof_torque[i * nb_cols + j] for all_dof_torque in log["all_dof_torque"]],
                        label="measured",
                    )
                    a.set(xlabel="time [s]", ylabel="Torque [Nm]", title=f"Joint Torque {i*nb_cols+j}")
                    a.legend()
                else:
                    break
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
