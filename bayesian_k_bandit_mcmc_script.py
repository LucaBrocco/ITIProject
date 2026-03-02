import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import warnings
import os

warnings.filterwarnings("ignore")
n_runs = 100
n_steps = 100
n_arms = 5

output_folder = f"simulation_results_{n_runs}_runs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

class GeneralMCMCBandit:
    def __init__(self, n_arms, n_samples=100):
        self.n_arms = n_arms
        self.n_samples = n_samples
        self.actions = [] 
        self.rewards = [] 
        self.posterior_mu = np.random.standard_t(df=3, size=(n_samples, n_arms))
        self.posterior_sigma = np.abs(np.random.normal(loc=1.0, scale=0.5, size=(n_samples, n_arms)))

    def select_arm(self):
        random_idx = np.random.randint(0, self.posterior_mu.shape[0])
        sample_means = self.posterior_mu[random_idx] 
        return np.argmax(sample_means)

    def update(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)
        obs_actions = np.array(self.actions) 
        obs_rewards = np.array(self.rewards)
        with pm.Model() as model:
            sigma = pm.HalfNormal('sigma', sigma=1.0, shape=self.n_arms) 
            mu = pm.StudentT('mu', nu=3, mu=0, sigma=2.0, shape=self.n_arms)
            mu_selected = mu[obs_actions] 
            sigma_selected = sigma[obs_actions]
            y = pm.Normal('y', mu=mu_selected, sigma=sigma_selected, observed=obs_rewards)
            idata = pm.sample(draws=self.n_samples, tune=1000, chains=1, progressbar=False, random_seed=42)
            extracted = az.extract(idata, var_names=['mu','sigma'])
            self.posterior_mu = extracted['mu'].values.T
            self.posterior_sigma = extracted['sigma'].values.T

def run_experiment(n_runs, n_steps, n_arms):
    all_run_regrets = []
    all_fraction_curves = []

    print(f"Starting simulation: {n_runs} runs, {n_steps} steps per run.")

    for run in range(n_runs):
        TRUE_MEANS = np.random.normal(-5, 5, n_arms)
        TRUE_SIGMAS = np.random.uniform(0.5, 2.5, n_arms)
        best_possible_mean = np.max(TRUE_MEANS)
        
        agent = GeneralMCMCBandit(n_arms=n_arms)
        optimal_action_history = []

        for t in range(n_steps):
            arm = agent.select_arm()
            reward = np.random.normal(loc=TRUE_MEANS[arm], scale=TRUE_SIGMAS[arm])
            agent.update(arm, reward) 
            is_optimal = 1 if TRUE_MEANS[arm] == best_possible_mean else 0
            optimal_action_history.append(is_optimal)

        timesteps = np.arange(1, n_steps + 1)
        optimal_fraction_curve = np.cumsum(optimal_action_history) / timesteps
        auc = np.trapz(optimal_fraction_curve, timesteps)
        regret_val = (n_steps - auc) / n_steps
        
        all_run_regrets.append(regret_val)
        all_fraction_curves.append(optimal_fraction_curve)

        # 1. Save/Append individual run regret
        with open(f"{output_folder}/all_run_regrets.csv", "a") as f:
            f.write(f"{regret_val}\n")

        # 2. Update Total (Current) Average Regret across all completed runs
        current_total_avg = np.mean(all_run_regrets)
        np.savetxt(f"{output_folder}/total_avg_regret.csv", [current_total_avg], delimiter=",")

        # 3. Save mean learning curve history (averaged across runs at each timestep)
        current_avg_fraction_curve = np.mean(all_fraction_curves, axis=0)
        np.savetxt(f"{output_folder}/mean_fraction_curve_history.csv", current_avg_fraction_curve, delimiter=",")
        
        if (run + 1) % 5 == 0 or (run + 1) == n_runs:
            print(f"Run {run+1}/{n_runs} finished. Total Avg Regret: {current_total_avg:.4f}")
            
            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            plt.plot(current_avg_fraction_curve, lw=3, color='blue')
            plt.fill_between(range(n_steps), current_avg_fraction_curve, alpha=0.2, color='blue')
            plt.title(f"Avg Learning Progress (Total Avg Regret: {current_total_avg:.4f})")
            plt.xlabel("Timesteps")
            plt.ylabel("Fraction of Optimal Choices")
            plt.ylim(0, 1.1)
            
            plt.subplot(1, 2, 2)
            plt.hist(all_run_regrets, color='teal', edgecolor='black', alpha=0.7)
            plt.axvline(current_total_avg, color='red', linestyle='--', label=f"Total Mean: {current_total_avg:.3f}")
            plt.title(f"Regret Distribution across {run+1} runs")
            plt.xlabel("Regret")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_folder}/progress_plot_run_{run+1}.png")
            plt.close()

    return all_run_regrets, all_fraction_curves

run_regrets, fraction_curves = run_experiment(n_runs=n_runs, n_steps=n_steps, n_arms=n_arms)
print(f"Done! Results stored in {output_folder}/")