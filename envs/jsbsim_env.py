import gymnasium as gym
import numpy as np

try:
    import jsbsim  # pip install jsbsim
except Exception as e:
    raise ImportError("JSBSim python package not found. Install with: pip install jsbsim") from e


def wrap_pi(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


class AeroPilotEnv(gym.Env):
    """
    AeroPilotEnv: RL Autopilot-style flight control environment using JSBSim.

    Modes (implicitly):
      - Altitude hold (ALT HOLD)
      - Airspeed hold (SPD HOLD)
      - Heading hold (HDG HOLD)

    Key upgrades vs minimal env:
      - Randomized setpoints per episode (commands)
      - Domain randomization (wind, mass, initial offsets)
      - Sensor noise + simple filtering
      - Action rate limiting (actuator realism)
      - Safety envelope checks (stall, overspeed, bank limits, min altitude)
      - Reward shaping with tolerance band bonus + smoothness penalty
      - Rich info metrics for evaluation
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        aircraft="c172p",
        dt=0.02,
        episode_seconds=90.0,
        # command ranges (commercial-style idea even if aircraft model is not airliner)
        alt_range_ft=(3000.0, 12000.0),
        spd_range_fps=(160.0, 320.0),
        hdg_change_rad=np.deg2rad(60.0),
        # safety envelope
        min_alt_ft=200.0,
        max_bank_rad=np.deg2rad(70.0),
        max_pitch_rad=np.deg2rad(30.0),
        min_vt_fps=90.0,
        max_vt_fps=500.0,
        max_aoa_rad=np.deg2rad(18.0),
        # noise/filter
        sensor_noise_std=None,
        lpf_alpha=0.35,
        # action rate limits (per step)
        max_delta_surface=0.08,  # per step change in elevator/aileron/rudder (normalized)
        max_delta_throttle=0.05,
        seed=0,
    ):
        super().__init__()
        self.dt = float(dt)
        self.max_steps = int(episode_seconds / self.dt)

        self.alt_range_ft = alt_range_ft
        self.spd_range_fps = spd_range_fps
        self.hdg_change_rad = float(hdg_change_rad)

        self.min_alt_ft = float(min_alt_ft)
        self.max_bank_rad = float(max_bank_rad)
        self.max_pitch_rad = float(max_pitch_rad)
        self.min_vt_fps = float(min_vt_fps)
        self.max_vt_fps = float(max_vt_fps)
        self.max_aoa_rad = float(max_aoa_rad)

        self.lpf_alpha = float(lpf_alpha)
        self.max_delta_surface = float(max_delta_surface)
        self.max_delta_throttle = float(max_delta_throttle)

        self.np_random = np.random.default_rng(seed)

        # Sensor noise setup
        # If not provided, set reasonable defaults
        if sensor_noise_std is None:
            sensor_noise_std = {
                "alt_ft": 3.0,
                "vt_fps": 1.5,
                "pitch_rad": np.deg2rad(0.15),
                "roll_rad": np.deg2rad(0.2),
                "psi_rad": np.deg2rad(0.25),
                "p_rad_s": np.deg2rad(0.3),
                "q_rad_s": np.deg2rad(0.3),
                "h_dot_fps": 0.8,
                "aoa_rad": np.deg2rad(0.2),
            }
        self.sensor_noise_std = sensor_noise_std

        # JSBSim init
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_dt(self.dt)

        if not self.fdm.load_model(aircraft):
            raise RuntimeError(
                f"Could not load aircraft model '{aircraft}'. "
                "If this fails, you need to point JSBSim to the aircraft data path."
            )

        # Actions: [elevator, aileron, rudder, throttle] continuous
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observations include:
        # core states + command errors + previous action (for smoothness learning)
        # [alt, vt, pitch, roll, psi, q, p, vs, aoa, alt_err, vt_err, hdg_err, prev_e, prev_a, prev_r, prev_t]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        self.steps = 0

        # Commands (targets)
        self.target_alt_ft = 5000.0
        self.target_vt_fps = 220.0
        self.target_psi_rad = 0.0

        # Filtering state (observations)
        self._obs_filt = None

        # previous action (for rate limiting + observation)
        self._prev_action = np.array([0.0, 0.0, 0.0, 0.6], dtype=np.float32)

    # ---------- JSBSim property access ----------
    def _raw_state(self):
        alt = float(self.fdm["position/h-sl-ft"])
        vt = float(self.fdm["velocities/vt-fps"])
        pitch = float(self.fdm["attitude/pitch-rad"])
        roll = float(self.fdm["attitude/roll-rad"])
        psi = float(self.fdm["attitude/psi-rad"])
        q = float(self.fdm["velocities/q-rad_sec"])
        p = float(self.fdm["velocities/p-rad_sec"])
        vs = float(self.fdm["velocities/h-dot-fps"])
        aoa = float(self.fdm.get_property_value("aero/alpha-rad")) if self.fdm.query_property("aero/alpha-rad") else 0.0
        return alt, vt, pitch, roll, psi, q, p, vs, aoa

    def _add_sensor_noise(self, alt, vt, pitch, roll, psi, q, p, vs, aoa):
        n = self.sensor_noise_std
        alt += self.np_random.normal(0.0, n["alt_ft"])
        vt += self.np_random.normal(0.0, n["vt_fps"])
        pitch += self.np_random.normal(0.0, n["pitch_rad"])
        roll += self.np_random.normal(0.0, n["roll_rad"])
        psi += self.np_random.normal(0.0, n["psi_rad"])
        q += self.np_random.normal(0.0, n["q_rad_s"])
        p += self.np_random.normal(0.0, n["p_rad_s"])
        vs += self.np_random.normal(0.0, n["h_dot_fps"])
        aoa += self.np_random.normal(0.0, n["aoa_rad"])
        return alt, vt, pitch, roll, psi, q, p, vs, aoa

    def _lpf(self, x):
        """Simple exponential low-pass filter."""
        if self._obs_filt is None:
            self._obs_filt = np.array(x, dtype=np.float32)
            return self._obs_filt.copy()

        alpha = self.lpf_alpha
        self._obs_filt = (1 - alpha) * self._obs_filt + alpha * np.array(x, dtype=np.float32)
        return self._obs_filt.copy()

    def _get_obs(self):
        alt, vt, pitch, roll, psi, q, p, vs, aoa = self._raw_state()
        alt, vt, pitch, roll, psi, q, p, vs, aoa = self._add_sensor_noise(
            alt, vt, pitch, roll, psi, q, p, vs, aoa
        )

        # command errors
        alt_err = alt - self.target_alt_ft
        vt_err = vt - self.target_vt_fps
        hdg_err = wrap_pi(psi - self.target_psi_rad)

        base = [alt, vt, pitch, roll, psi, q, p, vs, aoa, alt_err, vt_err, hdg_err]
        base = self._lpf(base)

        obs = np.concatenate([base, self._prev_action], axis=0).astype(np.float32)
        return obs

    # ---------- Domain randomization ----------
    def _randomize_initial_conditions(self):
        # Start near a nominal cruise-ish state, with random offsets
        alt0 = float(self.np_random.uniform(4500.0, 6500.0))
        vt0 = float(self.np_random.uniform(200.0, 260.0))
        pitch0 = float(self.np_random.normal(0.02, np.deg2rad(1.5)))
        roll0 = float(self.np_random.normal(0.0, np.deg2rad(2.0)))
        psi0 = float(self.np_random.uniform(-np.pi, np.pi))

        self.fdm["position/h-sl-ft"] = alt0
        self.fdm["velocities/vt-fps"] = vt0
        self.fdm["attitude/pitch-rad"] = pitch0
        self.fdm["attitude/roll-rad"] = roll0
        self.fdm["attitude/psi-rad"] = psi0

        # Optional: randomize fuel/weight if properties exist (not guaranteed across aircraft models)
        # Keep this defensive so it doesn't crash if the property isn't in the model.
        if self.fdm.query_property("inertia/weight-lbs"):
            w = self.fdm.get_property_value("inertia/weight-lbs")
            w *= float(self.np_random.uniform(0.95, 1.05))
            self.fdm.set_property_value("inertia/weight-lbs", w)

        # Simple wind randomization if model supports it (property names can vary)
        # Many JSBSim aircraft accept atmosphere/wind-north/east/down-fps, but not all.
        for prop, sigma in [
            ("atmosphere/wind-north-fps", 8.0),
            ("atmosphere/wind-east-fps", 8.0),
            ("atmosphere/wind-down-fps", 2.0),
        ]:
            if self.fdm.query_property(prop):
                self.fdm.set_property_value(prop, float(self.np_random.normal(0.0, sigma)))

    def _sample_commands(self):
        # Choose targets each episode (autopilot commands)
        self.target_alt_ft = float(self.np_random.uniform(*self.alt_range_ft))
        self.target_vt_fps = float(self.np_random.uniform(*self.spd_range_fps))

        # heading target is relative to current heading for realism
        psi_now = float(self.fdm["attitude/psi-rad"])
        delta = float(self.np_random.uniform(-self.hdg_change_rad, self.hdg_change_rad))
        self.target_psi_rad = wrap_pi(psi_now + delta)

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self._obs_filt = None

        self.fdm.reset_to_initial_conditions(0)

        # neutral controls
        self.fdm["fcs/elevator-cmd-norm"] = 0.0
        self.fdm["fcs/aileron-cmd-norm"] = 0.0
        self.fdm["fcs/rudder-cmd-norm"] = 0.0
        self.fdm["fcs/throttle-cmd-norm"] = 0.6
        self._prev_action = np.array([0.0, 0.0, 0.0, 0.6], dtype=np.float32)

        # randomize start + run IC
        self._randomize_initial_conditions()
        self.fdm.run_ic()

        # sample commands after IC (so it can be relative to current heading)
        self._sample_commands()

        obs = self._get_obs()
        info = {
            "target_alt_ft": self.target_alt_ft,
            "target_vt_fps": self.target_vt_fps,
            "target_psi_rad": self.target_psi_rad,
        }
        return obs, info

    def _rate_limit_action(self, action):
        action = np.array(action, dtype=np.float32)
        # clamp to bounds first
        action[0:3] = np.clip(action[0:3], -1.0, 1.0)
        action[3] = np.clip(action[3], 0.0, 1.0)

        # rate limits
        delta = action - self._prev_action
        delta[0:3] = np.clip(delta[0:3], -self.max_delta_surface, self.max_delta_surface)
        delta[3] = np.clip(delta[3], -self.max_delta_throttle, self.max_delta_throttle)

        new_action = self._prev_action + delta
        # final clamp
        new_action[0:3] = np.clip(new_action[0:3], -1.0, 1.0)
        new_action[3] = np.clip(new_action[3], 0.0, 1.0)
        return new_action

    def step(self, action):
        self.steps += 1

        # action with actuator realism
        a = self._rate_limit_action(action)
        elevator, aileron, rudder, throttle = map(float, a)

        self.fdm["fcs/elevator-cmd-norm"] = elevator
        self.fdm["fcs/aileron-cmd-norm"] = aileron
        self.fdm["fcs/rudder-cmd-norm"] = rudder
        self.fdm["fcs/throttle-cmd-norm"] = throttle

        ok = self.fdm.run()
        if not ok:
            obs = self._get_obs()
            return obs, -300.0, True, False, {"fail": "fdm_run_failed"}

        self._prev_action = a.copy()

        obs = self._get_obs()
        alt, vt, pitch, roll, psi, q, p, vs, aoa, alt_err, vt_err, hdg_err, *_ = obs

        # -------- reward shaping (control-inspired) --------
        # normalize errors for stable scaling
        alt_e = alt_err / 1000.0           # 1000 ft scale
        spd_e = vt_err / 100.0             # 100 fps scale
        hdg_e = hdg_err / np.deg2rad(30)   # 30 deg scale

        # tracking (L2-ish)
        track = (alt_e**2) + (0.8 * spd_e**2) + (0.3 * hdg_e**2)

        # stability penalties (rates + attitude)
        stab = 0.05 * (q**2 + p**2) + 0.02 * (pitch**2 + roll**2)

        # control effort + change penalty
        effort = 0.02 * (elevator**2 + aileron**2 + rudder**2) + 0.01 * (throttle - 0.6) ** 2

        reward = -(track + stab + effort)

        # bonus for staying within tolerance bands (encourages “lock-on”)
        in_alt = abs(alt_err) < 100.0
        in_spd = abs(vt_err) < 10.0
        in_hdg = abs(hdg_err) < np.deg2rad(5.0)
        if in_alt and in_spd and in_hdg:
            reward += 0.5

        # -------- safety envelope checks --------
        terminated = False
        fail_reason = None

        if alt < self.min_alt_ft:
            terminated = True
            reward -= 250.0
            fail_reason = "too_low"

        if vt < self.min_vt_fps:
            terminated = True
            reward -= 200.0
            fail_reason = "too_slow"

        if vt > self.max_vt_fps:
            terminated = True
            reward -= 200.0
            fail_reason = "overspeed"

        if abs(roll) > self.max_bank_rad:
            terminated = True
            reward -= 120.0
            fail_reason = "bank_limit"

        if abs(pitch) > self.max_pitch_rad:
            terminated = True
            reward -= 120.0
            fail_reason = "pitch_limit"

        if abs(aoa) > self.max_aoa_rad:
            terminated = True
            reward -= 250.0
            fail_reason = "stall_aoa"

        truncated = self.steps >= self.max_steps

        info = {
            "target_alt_ft": float(self.target_alt_ft),
            "target_vt_fps": float(self.target_vt_fps),
            "target_psi_rad": float(self.target_psi_rad),
            "alt_ft": float(alt),
            "vt_fps": float(vt),
            "psi_rad": float(psi),
            "alt_err_ft": float(alt_err),
            "vt_err_fps": float(vt_err),
            "hdg_err_rad": float(hdg_err),
            "aoa_rad": float(aoa),
            "fail_reason": fail_reason,
        }

        return obs, float(reward), terminated, truncated, info