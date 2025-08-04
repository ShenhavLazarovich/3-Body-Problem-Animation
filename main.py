
import numpy as np
from flask import Flask, render_template, jsonify, request
import json
import threading
import time
from datetime import datetime
from scipy.integrate import solve_ivp

app = Flask(__name__)

class ThreeBodySimulation:
    def __init__(self):
        # Gravitational constant (properly scaled for simulation units)
        self.G = 1.0  # Using normalized units for stability
        
        # Default initial conditions (positions and velocities)
        self.reset_to_default()
        
        # Simulation parameters
        self.dt = 0.01  # Time step for output
        self.time = 0
        self.running = False
        self.trajectory_length = 2000
        
        # Store trajectories for visualization
        self.trajectories = [[], [], []]
        
        # Integration method settings
        self.integrator_method = 'DOP853'  # High-order Runge-Kutta method
        self.rtol = 1e-9  # Relative tolerance
        self.atol = 1e-12  # Absolute tolerance
        
    def reset_to_default(self):
        """Reset to a 3D configuration that shows clear gravitational motion"""
        # Equal masses
        self.masses = np.array([1.0, 1.0, 1.0])
        
        # 3D initial conditions - positioned in different planes for true 3D motion
        self.positions = np.array([
            [1.0, 0.0, 0.5],      # Star 1 above xy-plane
            [-0.5, 0.866, -0.3],   # Star 2 below xy-plane 
            [-0.5, -0.866, 0.2]   # Star 3 slightly above xy-plane
        ])
        
        # 3D velocities for complex orbital motion
        self.velocities = np.array([
            [0.2, 0.5, 0.3],      # Star 1 with z-component
            [-0.4, 0.1, -0.2],    # Star 2 with negative z-component
            [0.2, -0.6, 0.4]      # Star 3 with positive z-component
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
    
    def three_body_ode(self, t, y):
        """
        ODE system for the three-body problem
        y contains [x1, y1, z1, x2, y2, z2, x3, y3, z3, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
        """
        # Extract positions and velocities
        pos = y[:9].reshape(3, 3)  # positions of 3 bodies
        vel = y[9:].reshape(3, 3)  # velocities of 3 bodies
        
        # Calculate accelerations
        acc = np.zeros_like(pos)
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    # Vector from body i to body j
                    r_vec = pos[j] - pos[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    # Avoid singularity with small softening parameter
                    if r_mag > 1e-8:
                        # Gravitational acceleration
                        acc[i] += self.G * self.masses[j] * r_vec / (r_mag**3)
        
        # Return derivatives: velocities and accelerations
        dydt = np.concatenate([vel.flatten(), acc.flatten()])
        return dydt
    
    def step(self):
        """Perform one integration step using scipy's advanced ODE solvers"""
        if not self.running:
            return
        
        # Prepare initial conditions for ODE solver
        y0 = np.concatenate([self.positions.flatten(), self.velocities.flatten()])
        
        # Time span for this step
        t_span = (self.time, self.time + self.dt)
        t_eval = [self.time + self.dt]
        
        try:
            # Solve ODE using high-precision method
            sol = solve_ivp(
                self.three_body_ode, 
                t_span, 
                y0, 
                method=self.integrator_method,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=False
            )
            
            if sol.success and len(sol.y) > 0:
                # Extract new positions and velocities
                y_new = sol.y[:, -1]
                self.positions = y_new[:9].reshape(3, 3)
                self.velocities = y_new[9:].reshape(3, 3)
                
                # Store trajectories
                for i in range(3):
                    if len(self.trajectories[i]) >= self.trajectory_length:
                        self.trajectories[i].pop(0)
                    self.trajectories[i].append(self.positions[i].tolist())
                
                self.time += self.dt
            else:
                print(f"ODE solver failed: {sol.message}")
                
        except Exception as e:
            print(f"Integration error: {e}")
            # Fall back to simple method if scipy fails
            self.simple_step()
    
    def simple_step(self):
        """Fallback simple integration method"""
        # Calculate forces
        forces = np.zeros_like(self.positions)
        
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_vec = self.positions[j] - self.positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    if r_mag > 1e-8:
                        force_mag = self.G * self.masses[i] * self.masses[j] / (r_mag**2)
                        force_dir = r_vec / r_mag
                        forces[i] += force_mag * force_dir
        
        # Update using Verlet integration
        accelerations = forces / self.masses.reshape(-1, 1)
        
        # Velocity-Verlet method
        self.positions += self.velocities * self.dt + 0.5 * accelerations * self.dt**2
        self.velocities += accelerations * self.dt
        
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
        time.sleep(0.005)  # 200 FPS simulation for smooth motion

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
