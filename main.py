
import numpy as np
from flask import Flask, render_template, jsonify, request
import json
import threading
import time
from datetime import datetime

app = Flask(__name__)

class ThreeBodySimulation:
    def __init__(self):
        # Gravitational constant (properly scaled for simulation units)
        self.G = 1.0  # Using normalized units for stability
        
        # Default initial conditions (positions and velocities)
        self.reset_to_default()
        
        # Simulation parameters
        self.dt = 0.005  # Slightly larger time step for better stability
        self.time = 0
        self.running = False
        self.trajectory_length = 1000
        
        # Store trajectories for visualization
        self.trajectories = [[], [], []]
        
    def reset_to_default(self):
        """Reset to a stable figure-8 configuration that shows clear gravitational motion"""
        # Equal masses
        self.masses = np.array([1.0, 1.0, 1.0])
        
        # Figure-8 initial conditions (classic three-body solution)
        self.positions = np.array([
            [-0.97000436, 0.24308753, 0.0],
            [0.97000436, -0.24308753, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        # Carefully chosen velocities for stable motion
        self.velocities = np.array([
            [0.4662036850, 0.4323657300, 0.0],
            [0.4662036850, 0.4323657300, 0.0],
            [-0.93240737, -0.86473146, 0.0]
        ])
        
        self.time = 0
        self.trajectories = [[], [], []]
    
    def set_initial_conditions(self, positions, velocities, masses):
        """Set custom initial conditions"""
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.masses = np.array(masses)
        self.time = 0
        self.trajectories = [[], [], []]
    
    def compute_forces(self):
        """Compute gravitational forces between all bodies"""
        forces = np.zeros_like(self.positions)
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    # Vector from i to j
                    r_vec = self.positions[j] - self.positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    # Avoid singularity
                    if r_mag > 1e-10:
                        # Gravitational force
                        force_mag = self.G * self.masses[i] * self.masses[j] / (r_mag**3)
                        forces[i] += force_mag * r_vec
        
        return forces
    
    def step(self):
        """Perform one integration step using Runge-Kutta 4th order"""
        if not self.running:
            return
            
        # RK4 integration for better accuracy
        def derivatives(pos, vel):
            forces = np.zeros_like(pos)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        r_vec = pos[j] - pos[i]
                        r_mag = np.linalg.norm(r_vec)
                        # Add softening parameter to prevent singularities
                        if r_mag > 1e-6:
                            # Correct gravitational force calculation
                            force_mag = self.G * self.masses[i] * self.masses[j] / (r_mag**2)
                            force_dir = r_vec / r_mag
                            forces[i] += force_mag * force_dir
            
            accelerations = forces / self.masses.reshape(-1, 1)
            return vel, accelerations
        
        # RK4 steps
        k1_pos, k1_vel = derivatives(self.positions, self.velocities)
        k2_pos, k2_vel = derivatives(self.positions + 0.5*self.dt*k1_pos, 
                                   self.velocities + 0.5*self.dt*k1_vel)
        k3_pos, k3_vel = derivatives(self.positions + 0.5*self.dt*k2_pos, 
                                   self.velocities + 0.5*self.dt*k2_vel)
        k4_pos, k4_vel = derivatives(self.positions + self.dt*k3_pos, 
                                   self.velocities + self.dt*k3_vel)
        
        # Update positions and velocities
        self.positions += self.dt/6 * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        self.velocities += self.dt/6 * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
        
        # Store trajectories
        for i in range(3):
            if len(self.trajectories[i]) >= self.trajectory_length:
                self.trajectories[i].pop(0)
            self.trajectories[i].append(self.positions[i].tolist())
        
        self.time += self.dt
    
    def get_state(self):
        """Get current simulation state"""
        return {
            'positions': self.positions.tolist(),
            'velocities': self.velocities.tolist(),
            'masses': self.masses.tolist(),
            'time': self.time,
            'trajectories': self.trajectories,
            'running': self.running
        }

# Global simulation instance
simulation = ThreeBodySimulation()

def simulation_loop():
    """Main simulation loop running in background thread"""
    while True:
        if simulation.running:
            simulation.step()
        time.sleep(0.01)  # 100 FPS simulation

# Start simulation thread
simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
simulation_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    return jsonify(simulation.get_state())

@app.route('/api/start', methods=['POST'])
def start_simulation():
    simulation.running = True
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    simulation.running = False
    return jsonify({'status': 'stopped'})

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    simulation.running = False
    simulation.reset_to_default()
    return jsonify({'status': 'reset'})

@app.route('/api/set_initial', methods=['POST'])
def set_initial_conditions():
    data = request.json
    try:
        positions = data['positions']
        velocities = data['velocities']
        masses = data['masses']
        
        simulation.running = False
        simulation.set_initial_conditions(positions, velocities, masses)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
