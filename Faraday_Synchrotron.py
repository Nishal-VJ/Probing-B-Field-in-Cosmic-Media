import numpy as bro
import matplotlib.pyplot as nishal
from math import gamma

# Physical constants (cgs)
e   = 4.8032068e-10 #statC = esu 
m_e = 9.1093837e-28 # g 
c_light   = 2.99792458e10 # cm/s

bro.random.seed(42)

# ============================================================================
# SYNCHROTRON EMISSION CONSTANTS
# ============================================================================
alpha = 0.5 

# Synchrotron emissivity prefactor (from Rybicki & Lightman)
epsilon_alpha_I = (3**(alpha + 0.5) / (4 * (alpha + 1))) * gamma(alpha/2 + 11/6) * gamma(alpha/2 + 1/6)
print(f"Synchrotron emissivity prefactor epsilon_alpha_I = {epsilon_alpha_I:.6e}")

# Particle density / magnetic field scaling (you calibrate this)
# Higher values = brighter synchrotron
epsilon_perp = 1.0e-20  

fractional_polarization = 0.25  # Synchrotron is ~70% polarized for alpha=0.5

# ============================================================================
# SETUP: Grid, fields, initial conditions
# ============================================================================

# observing wavelength [m]
lambda_m = 0.21 

# background radiation (initial condition) at source
I_source = 1.0  # Intensity from the source
Q_source = 0.7 * I_source  # Some initial linear polarization
U_source = 0.0

# Distance grid from observer (0) to source (L)
L_pc  = 1000.0
Nz    = 2000
s_pc  = bro.linspace(0.0, L_pc, Nz)

# NON UNIFORM ELECTRON DENSITY (for Faraday rotation)
n_e_raw = bro.random.lognormal(0.0, 1.0, Nz)
n_e_th = 0.03 * (n_e_raw / bro.mean(n_e_raw))

# MAGNETIC FIELD VECTOR [microGauss]
# Tangled field with some structure
Bx = 2.0 * bro.sin(2*bro.pi*s_pc/L_pc) + 0.5*bro.random.randn(Nz)
By = 2.0 * bro.cos(2*bro.pi*s_pc/L_pc) + 0.5*bro.random.randn(Nz)
Bz = 3.0*bro.ones_like(s_pc) + 0.3*bro.random.randn(Nz)

# Line-of-sight unit vector
los = bro.array([0.,0.,1.])

# unit conversion
lambda_cm = lambda_m*100.0
s_cm      = s_pc*3.08567758e18

# Construction of B_parallel (for Faraday rotation coefficient)
B_parallel_uG = Bx*los[0] + By*los[1] + Bz*los[2]
B_parallel_G  = B_parallel_uG*1e-6

# Faraday rotation coefficient function
def faraday_rotation(n_e, B_par, lam_cm):
    f_th = (1/bro.pi)*(e**3/(m_e**2*c_light**4))*n_e*B_par*lam_cm**2
    return f_th

# RM in standard astronomical units
RM = 0.812*bro.trapezoid(n_e_th*B_parallel_uG, s_pc)
print(f'RM = {RM:.3f} rad m^-2')

# ============================================================================
# FREQUENCY RANGE AND TABLES
# ============================================================================

freq_range = [15e9, 1.4e9, 350e6]

# Initialize 2D tables for I, Q, U
I_table = bro.zeros((len(freq_range), Nz))
Q_table = bro.zeros((len(freq_range), Nz))
U_table = bro.zeros((len(freq_range), Nz))

# ============================================================================
# MAIN RK4 INTEGRATION LOOP
# ============================================================================

for idx, freq_value in enumerate(freq_range):
    print(f"\n--- Processing frequency {freq_value/1e9:.1f} GHz ---")
    
    # Wavelength in cm
    lamda_cm = c_light / freq_value
    omega = 2 * bro.pi * freq_value
    
    # Faraday rotation coefficient (may vary with position)
    f_th = faraday_rotation(n_e_th, B_parallel_G, lamda_cm)
    
    # Set boundary conditions at the SOURCE (z = Nz-1)
    # I_table[idx, Nz-1] = I_source
    Q_table[idx, Nz-1] = Q_source
    U_table[idx, Nz-1] = U_source
    
    # ========================================================================
    # INNER LOOP: integrate from source back to observer
    # ========================================================================
    for i in range(Nz-1, 0, -1):
        ds = s_cm[i] - s_cm[i-1]  # Physical step size in cm
        f = f_th[i]                # Local Faraday rotation coefficient
        
        # --- SYNCHROTRON EMISSION CALCULATION FOR THIS CELL ---
        
        # Total magnetic field strength
        B_tot_uG = bro.sqrt(Bx[i]**2 + By[i]**2 + Bz[i]**2)
        B_tot_G = B_tot_uG * 1e-6
        
        # Angle between B-field and line of sight
        cos_theta = bro.dot([Bx[i], By[i], Bz[i]], los) / (B_tot_uG + 1e-10)
        cos_theta = bro.clip(cos_theta, -1.0, 1.0)
        theta = bro.arccos(cos_theta)
        
        # Perpendicular component of gyrofrequency
        omega_B_perp = (e * B_tot_G / (m_e * c_light)) * bro.sin(theta)
        
        # Synchrotron intensity emissivity (Eq. from your notes)
        # epsilon_I,nt = epsilon_alpha_I * epsilon_perp * (omega_B_perp / omega)^alpha
        epsilon_I = epsilon_alpha_I * epsilon_perp * (omega_B_perp / omega)**alpha
        
        # Projection angle: polarization is perpendicular to B-field on sky
        # B_perp = (Bx, By) component on the sky
        phi_sky = bro.arctan2(By[i], Bx[i])
        chi_0 = phi_sky + (bro.pi / 2.0)  # Polarization angle (perpendicular to B)
        
        # Project emission into Q and U Stokes components
        # (synchrotron is linearly polarized perpendicular to B)
        epsilon_Q = epsilon_I * fractional_polarization * bro.cos(2 * chi_0)
        epsilon_U = epsilon_I * fractional_polarization * bro.sin(2 * chi_0)
        
        # --- RK4 INTEGRATION WITH SOURCE TERMS ---
        # We're solving: 
        #   dI/ds = epsilon_I
        #   dQ/ds = f*U + epsilon_Q
        #   dU/ds = -f*Q + epsilon_U
        
        I_old = I_table[idx, i]
        Q_old = Q_table[idx, i]
        U_old = U_table[idx, i]
        
        # k1 (at current point)
        k1_I =  epsilon_I * ds
        k1_Q = (f * U_old + epsilon_Q) * ds
        k1_U = (-f * Q_old + epsilon_U) * ds
        
        # k2 (midpoint, using k1 estimates)
        # Note: we're holding f, epsilon constant over this step
        # (you could interpolate if you wanted finer resolution)
        k2_I =  epsilon_I * ds
        k2_Q = (f * (U_old + 0.5*k1_U) + epsilon_Q) * ds
        k2_U = (-f * (Q_old + 0.5*k1_Q) + epsilon_U) * ds
        
        # k3 (midpoint, using k2 estimates)
        k3_I =  epsilon_I * ds
        k3_Q = (f * (U_old + 0.5*k2_U) + epsilon_Q) * ds
        k3_U = (-f * (Q_old + 0.5*k2_Q) + epsilon_U) * ds
        
        # k4 (at next point, using k3 estimates)
        k4_I =  epsilon_I * ds
        k4_Q = (f * (U_old + k3_U) + epsilon_Q) * ds
        k4_U = (-f * (Q_old + k3_Q) + epsilon_U) * ds
        
        # RK4 weighted average
        I_table[idx, i-1] = I_old + (1.0/6.0) * (k1_I + 2*k2_I + 2*k3_I + k4_I)
        Q_table[idx, i-1] = Q_old + (1.0/6.0) * (k1_Q + 2*k2_Q + 2*k3_Q + k4_Q)
        U_table[idx, i-1] = U_old + (1.0/6.0) * (k1_U + 2*k2_U + 2*k3_U + k4_U)
    
    # Check conservation
    P2_at_obs = Q_table[idx, 0]**2 + U_table[idx, 0]**2
    print(f"  At observer: I={I_table[idx, 0]:.6f}, Q={Q_table[idx, 0]:.6f}, U={U_table[idx, 0]:.6f}")
    print(f"  Polarized intensity P = sqrt(Q²+U²) = {bro.sqrt(P2_at_obs):.6f}")

# ==================================================================================================================================================
# VISUALIZATION
# ==================================================================================================================================================

fig, axes = nishal.subplots(3, 1, figsize=(12, 10))

freq_labels = ['15 GHz', '1.4 GHz', '350 MHz']
colors = ['blue', 'orange', 'green']

# --- PANEL 1: INTENSITY (I) ---
for idx, (freq, color, label) in enumerate(zip(freq_range, colors, freq_labels)):
    axes[0].plot(s_pc, I_table[idx, :], label=label, color=color, linewidth=2)
axes[0].set_xlabel('Distance along LOS (parsecs)')
axes[0].set_ylabel('Intensity I')
axes[0].set_title('Total Intensity Evolution (Synchrotron Emission)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, L_pc)

# --- PANEL 2: LINEAR POLARIZATION (Q) ---
for idx, (freq, color, label) in enumerate(zip(freq_range, colors, freq_labels)):
    axes[1].plot(s_pc, Q_table[idx, :], label=label, color=color, linewidth=2)
axes[1].set_xlabel('Distance along LOS (parsecs)')
axes[1].set_ylabel('Stokes Q')
axes[1].set_title('Linear Polarization Q (Faraday Rotation + Emission)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, L_pc)

# --- PANEL 3: LINEAR POLARIZATION (U) ---
for idx, (freq, color, label) in enumerate(zip(freq_range, colors, freq_labels)):
    axes[2].plot(s_pc, U_table[idx, :], label=label, color=color, linewidth=2)
axes[2].set_xlabel('Distance along LOS (parsecs)')
axes[2].set_ylabel('Stokes U')
axes[2].set_title('Linear Polarization U (Faraday Rotation + Emission)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, L_pc)

nishal.tight_layout()
nishal.show()

# ============================================================================
# POLARIZATION ANALYSIS
# ============================================================================

fig2, axes2 = nishal.subplots(1, 2, figsize=(12, 5))

# --- Polarized intensity P = sqrt(Q^2 + U^2) ---
for idx, (freq, color, label) in enumerate(zip(freq_range, colors, freq_labels)):
    P = bro.sqrt(Q_table[idx, :]**2 + U_table[idx, :]**2)
    axes2[0].plot(s_pc, P, label=label, color=color, linewidth=2)
axes2[0].set_xlabel('Distance along LOS (parsecs)')
axes2[0].set_ylabel('Polarized Intensity P')
axes2[0].set_title('Polarized Intensity P = √(Q² + U²)')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xlim(0, L_pc)

# --- Polarization angle chi = 0.5 * arctan2(U, Q) ---
for idx, (freq, color, label) in enumerate(zip(freq_range, colors, freq_labels)):
    chi = 0.5 * bro.arctan2(U_table[idx, :], Q_table[idx, :])
    axes2[1].plot(s_pc, bro.degrees(chi), label=label, color=color, linewidth=2)
axes2[1].set_xlabel('Distance along LOS (parsecs)')
axes2[1].set_ylabel('Polarization Angle χ (degrees)')
axes2[1].set_title('Polarization Angle χ')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)
axes2[1].set_xlim(0, L_pc)

nishal.tight_layout()
nishal.show()

# ============================================================================
# SUMMARY TABLE AT OBSERVER
# ============================================================================

print("\n" + "="*70)
print("VALUES AT OBSERVER (s=0):")
print("="*70)
print(f"{'Frequency':<12} {'I':<12} {'Q':<12} {'U':<12} {'P':<12} {'χ (°)':<12}")
print("-"*70)
for idx, freq in enumerate(freq_range):
    I_obs = I_table[idx, 0]
    Q_obs = Q_table[idx, 0]
    U_obs = U_table[idx, 0]
    P_obs = bro.sqrt(Q_obs**2 + U_obs**2)
    chi_obs = 0.5 * bro.arctan2(U_obs, Q_obs)
    print(f"{freq/1e9:>6.1f} GHz  {I_obs:>11.5f} {Q_obs:>11.5f} {U_obs:>11.5f} {P_obs:>11.5f} {bro.degrees(chi_obs):>11.1f}")
