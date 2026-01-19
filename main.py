
import taichi as ti
import math
import os

ti.init(arch=ti.gpu)

#Constante
G = 1
c =1

#Parametre de la simulation
dt = 1
num_p = 1500000
screen = [1980,1200]

F_img= "frames"
frame_T = 5000
os.makedirs(F_img, exist_ok=True)

ch_E = ti.Vector([0,0,0])
ch_B = ti.Vector([0,0,10])


#Parametre du trou noir
M = 1
a = 1
r_s = (2 * G * M) / c**2
r_h = (r_s + math.sqrt(r_s**2 - 4 * a**2)) / 2
T_H = 1 / (8 * math.pi * M)
pos_trou_noir = ti.Vector.field(3, dtype=ti.f32, shape=1)
pos_trou_noir[0] = ti.Vector([0.0, 0.0, 0.0])


#Parametre de la particule
E = ti.field(dtype=float, shape=(num_p))
L = ti.field(dtype=float, shape=(num_p))
Q = ti.field(dtype=float, shape=(num_p))
Temp = ti.field(dtype=float, shape=(num_p))
q = ti.field(dtype=float, shape=(num_p))
mu = 1

r = ti.field(dtype=float, shape=(num_p))
theta = ti.field(dtype=float, shape=(num_p))
phi = ti.field(dtype=float, shape=(num_p))

signe_r = ti.field(dtype=int, shape=(num_p))
signe_theta = ti.field(dtype=int, shape=(num_p))

pos = ti.Vector.field(3, dtype=float, shape=(num_p))
color = ti.Vector.field(4, dtype=float, shape=(num_p))


@ti.func
def get_delta(r_val):
    return r_val**2 - r_s * r_val + a**2
@ti.func
def get_sigma(r_val, theta_val):
    return r_val**2 + a**2 * ti.cos(theta_val)**2
@ti.func
def get_P(r_val,E_val,L_val):
    return E_val * (r_val**2 + a**2) - a * L_val
@ti.func
def get_R(r_val,E_val,L_val,Q_val):
    return get_P(r_val,E_val,L_val)**2 - get_delta(r_val) * (mu**2 * r_val**2 + (L_val - a * E_val)**2 + Q_val)
@ti.func
def get_theta(r_val,theta_val,E_val,L_val,Q_val):
    return Q_val - ti.cos(theta_val)**2 * (a**2 * (mu**2 - E_val**2) + ((L_val**2 )/ ti.sin(theta_val)**2))

@ti.func
def get_A(r_val,theta_val):
    return (r_val**2 + a**2)**2 - a**2 * get_delta(r_val) * ti.sin(theta_val)**2

@ti.func
def get_lapse(r_val,theta_val):
    return ti.sqrt(get_delta(r_val)*get_sigma(r_val, theta_val)/get_A(r_val,theta_val))
@ti.func
def get_shift(r_val,theta_val):
    return (2 * M * a * r_val)/get_A(r_val,theta_val)
@ti.func
def get_gamma(r_val,theta_val):
    gamma_rr = get_sigma(r_val,theta_val)/get_delta(r_val)
    gamma_theta = get_sigma(r_val,theta_val)
    gamma_phi = (get_A(r_val,theta_val) * ti.sin(theta_val)**2) / get_sigma(r_val,theta_val)
    return gamma_rr, gamma_theta, gamma_phi
@ti.func
def get_time(r_val,theta_val,E_val,L_val):
    terme1 = get_A(r_val,theta_val) * E_val - 2 * a * M * r_val * L_val
    terme2 = get_delta(r_val) * get_sigma(r_val,theta_val)
    return terme1 / terme2

#get_dérivées_pos
@ti.func
def get_dr(r_val,theta_val,E_val,L_val,Q_val,signe_r_val):
    val_R = get_R(r_val,E_val,L_val,Q_val)
    return (signe_r_val * ti.sqrt(ti.max(0.0, val_R)) / get_sigma(r_val, theta_val))

@ti.func
def get_dtheta(r_val,theta_val,E_val,L_val,Q_val,signe_theta_val):
    val_theta = get_theta(r_val,theta_val,E_val,L_val,Q_val)
    return (signe_theta_val * ti.sqrt(ti.max(0.0,val_theta)) / get_sigma(r_val, theta_val))

@ti.func
def get_dphi(r_val,theta_val,E_val,L_val):
    terme1 = -(a * E_val - (L_val/ti.sin(theta_val)**2))
    terme2 = get_P(r_val,E_val,L_val) *  (a/get_delta(r_val))
    return (terme1 + terme2 )/ get_sigma(r_val, theta_val)

@ti.func
def get_v(r_val,theta_val,phi_val,E_val,L_val,dr_dt,dtheta_dt,dphi_dt):
    dt_dt= get_time(r_val,theta_val,E_val,L_val)
    terme1 = get_gamma(r_val,theta_val)[0] * dr_dt**2
    terme2 = get_gamma(r_val, theta_val)[1] * dtheta_dt**2
    terme3 = get_gamma(r_val, theta_val)[2] * (dphi_dt - get_shift(r_val,theta_val) * get_time(r_val,theta_val,E_val,L_val))**2
    terme4 = (get_lapse(r_val,theta_val) * get_time(r_val,theta_val,E_val,L_val))**2
    return ti.sqrt((terme1 + terme2 + terme3) / terme4)


@ti.func
def get_temp(r_val,theta_val,phi_val,E_val,L_val,dr_dt,dtheta_dt,dphi_dt):
    #T_r = T_H / (ti.sqrt(ti.max(0.0, 1 - (r_s / r_val))) + 1e-5)
    v = get_v(r_val,theta_val,phi_val,E_val,L_val,dr_dt,dtheta_dt,dphi_dt)
    gamma = 1.0 / ti.sqrt(1- v**2)
    return T_H * gamma


#Pas fait par moi (temp_to_color) pour les codes couleurs
@ti.func
def temp_to_color(temp_val) -> ti.types.vector(4,float):
    c_noir = ti.Vector([0.0, 0.0, 0.0,1])
    c_rouge_fonce = ti.Vector([0.3, 0.0, 0.0,1])  # 0.1 : Infrarouge profond
    c_rouge = ti.Vector([0.8, 0.1, 0.0,1])  # 0.2 : Rouge chaud
    c_orange = ti.Vector([1.0, 0.4, 0.0,100])  # 0.35 : Orange
    c_jaune = ti.Vector([1.0, 0.9, 0.1,1])  # 0.5 : Jaune
    c_blanc = ti.Vector([1.0, 1.0, 1.0,0.9])  # 0.7 : Blanc pur
    c_bleu_clair = ti.Vector([0.4, 0.7, 1.0, 0.9])  # 0.85 : Bleu électrique
    c_violet = ti.Vector([0.6, 0.2, 1.0, 8 ])  # 1.0+ : Rayons X / Ultra-violet

    res = c_noir

    if temp_val < 0.1:
        t = temp_val / 0.1
        res = c_noir * (1.0 - t) + c_rouge_fonce * t

    elif temp_val < 0.2:
        t = (temp_val - 0.1) / 0.1
        res = c_rouge_fonce * (1.0 - t) + c_rouge * t

    elif temp_val < 0.35:
        t = (temp_val - 0.2) / 0.15
        res = c_rouge * (1.0 - t) + c_orange * t

    elif temp_val < 0.5:
        t = (temp_val - 0.35) / 0.15
        res = c_orange * (1.0 - t) + c_jaune * t

    elif temp_val < 0.7:
        t = (temp_val - 0.5) / 0.2
        res = c_jaune * (1.0 - t) + c_blanc * t

    elif temp_val < 0.8:
        t = (temp_val - 0.7) / 0.2
        res = c_blanc * (1.0 - t) + c_bleu_clair * t

    else:
        t = ti.min(1.0, (temp_val - 0.9) / 0.5)
        res = c_bleu_clair * (1.0 - t) + c_violet * t

    return res

@ti.func
def get_Lorentz(r_val,theta_val,phi_val,dr_dt,dtheta_dt,dphi_dt,q_val):
    B_r = 5000.5 * ((2*ti.cos(theta_val))/ r_val**3)
    B_theta = 5.5 * (ti.sin(theta_val) / r_val**3)
    B_phi = 200 * (1/r_val**2)

    F_lorentz_r =  q_val * ((dtheta_dt * r_val) * B_phi - (r_val * dphi_dt * ti.sin(theta_val)) * B_theta)
    F_lorentz_theta = q_val * ((r_val * dphi_dt * ti.sin(theta_val)) * B_r - dr_dt * B_phi)
    F_lorentz_phi = q_val * (dr_dt * B_theta - (dtheta_dt * r_val) * B_r)
    return F_lorentz_r, F_lorentz_theta, F_lorentz_phi

@ti.func
def get_F_f(dr_dt,dtheta_dt,dphi_dt,q_val):
    F_f_r = -0.5* dr_dt
    F_f_theta =-0.5 * dtheta_dt
    F_f_phi = -0.5 * dphi_dt
    return F_f_r, F_f_theta, F_f_phi


@ti.kernel
def place_p():
    for i in range(num_p):
        r[i] = r_h + 200 + ti.random() * 250
        phi[i] = ti.random() * 2 * math.pi *50
        theta[i] = math.pi/2 + (ti.random() - 0.5) * 0.5

        E[i] = 0.5 + ti.random() * 0.75
        L[i] = 20 + ti.random() * 5
        Q[i] =  ti.random() * 0.2

        signe_r[i] = -1
        signe_theta[i] = -1

        if ti.random() < 0.5:
            q[i] = 1
        else:
            q[i] = -1

        # traduction en [x,y,z]...
        x = ti.sqrt(r[i] ** 2 + a ** 2) * ti.sin(theta[i]) * ti.cos(phi[i])
        y = ti.sqrt(r[i] ** 2 + a ** 2) * ti.sin(theta[i]) * ti.sin(phi[i])
        z = r[i] * ti.cos(theta[i])
        pos[i] = ti.Vector([x, z, y])

@ti.kernel
def update_p():
    for i in range(num_p):
        dr0 = r[i]
        dtheta0 = theta[i]
        dphi0 = phi[i]
        q_val = q[i]

        dr_dt, dtheta_dt, dphi_dt = get_dr(dr0,dtheta0,E[i],L[i],Q[i],signe_r[i]), get_dtheta(dr0,dtheta0,E[i],L[i],Q[i],signe_theta[i]), get_dphi(dr0,dtheta0,E[i],L[i])

        F_l_r, F_l_theta,F_l_phi= get_Lorentz(dr0,dtheta0,dphi0,dr_dt,dtheta_dt,dphi_dt,q_val)
        F_f_r, F_f_theta, F_f_phi = get_F_f(dr_dt,dtheta_dt,dphi_dt,q_val)

        Temp= get_temp(dr0, dtheta0, dphi0,E[i],L[i],dr_dt,dtheta_dt,dphi_dt)
        color[i] = temp_to_color(Temp*8)

        if dr0 <= r_h*2.6 or dr0 >= 800:
            dr0 = r_h + 200 + ti.random() * 250
            dphi0 = ti.random() * 2 * math.pi
            dtheta0 = math.pi / 2 + (ti.random() - 0.5) * 0.5

            E[i] = 0.5 + ti.random() * 0.75
            L[i] = 20 + ti.random() * 5
            Q[i] = ti.random() * 0.2

            signe_r[i] = -1
            signe_theta[i] = -1

            if ti.random() < 0.5:
                q[i] = -1
            else:
                q[i] = +1

        if theta[i] < 0.15 or theta[i] > math.pi - 0.15:
            F_l_r *= 1
            F_l_theta *= 1
            F_l_phi *= 1
        else:
            F_l_r *= 0.5
            F_l_theta *= 0.5
            F_l_phi *= 0.5

        #k1
        k1_r,k1_t,k1_p = get_dr(dr0,dtheta0,E[i],L[i],Q[i],signe_r[i]) + F_l_r + F_f_r, get_dtheta(dr0,dtheta0,E[i],L[i],Q[i],signe_theta[i]) + F_l_theta + F_f_theta, get_dphi(dr0,dtheta0,E[i],L[i]) + F_l_phi + F_f_phi

        #k2---
        dr1 = dr0 + 0.5 * dt * k1_r
        dtheta1 = dtheta0 + 0.5 * dt * k1_t
        dphi1 = dphi0 + 0.5 * dt * k1_p
        k2_r, k2_t, k2_p = get_dr(dr1, dtheta1,E[i],L[i],Q[i],signe_r[i]), get_dtheta(dr1, dtheta1,E[i],L[i],Q[i],signe_theta[i]), get_dphi(dr1, dtheta1,E[i],L[i])
        dr2 = dr1 + 0.5 * dt * k2_r

        #k3-----
        dr2 = dr0 + 0.5 * dt * k2_r
        dtheta2 = dtheta0 + 0.5 * dt * k2_t
        dphi2 = dphi0 + 0.5 * dt * k2_p
        k3_r, k3_t, k3_p = get_dr(dr2, dtheta2,E[i],L[i],Q[i],signe_r[i]), get_dtheta(dr2, dtheta2,E[i],L[i],Q[i],signe_theta[i]), get_dphi(dr2, dtheta2,E[i],L[i])

        #k4-----
        dr3 = dr0 + dt * k3_r
        dtheta3 = dtheta0 + dt * k3_t
        dphi3 = dphi0 + dt * k3_p
        k4_r, k4_t, k4_p = get_dr(dr3, dtheta3,E[i],L[i],Q[i],signe_r[i]), get_dtheta(dr3, dtheta3,E[i],L[i],Q[i],signe_theta[i]), get_dphi(dr3, dtheta3,E[i],L[i])

        #final---
        r[i] = dr0 + dt * (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
        theta[i] = dtheta0 + dt * (k1_t + 2*k2_t + 2*k3_t + k4_t) / 6  #+(0.001 * ti.cos(theta[i]))
        phi[i] = dphi0 + dt * (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6

        if r[i] > r_h + 1:
            if get_R(r[i], E[i], L[i], Q[i]) < 0:
                signe_r[i] *= -1
                #r[i] += signe_r[i] * -1 * dt * 0.1
                #r[i] = dr0
                #r[i] += 0.5

            if get_theta(r[i], theta[i], E[i], L[i], Q[i]) < 0:
                signe_theta[i] *= -1


        #traduction en [x,y,z]...
        x = ti.sqrt(r[i]**2 + a**2) * ti.sin(theta[i]) * ti.cos(phi[i])
        y = ti.sqrt(r[i]**2 + a**2) * ti.sin(theta[i]) * ti.sin(phi[i])
        z = r[i] * ti.cos(theta[i])
        pos[i] = ti.Vector([x, z,y])


window = ti.ui.Window(f"BlackHoleKerr avec {num_p} particules", (screen[0], screen[1]), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0, 500, 500)
camera.lookat(0, 0, 0)

gg = (1000, 1000)

img = ti.Vector.field(3, ti.f32, shape=gg)
bright = ti.Vector.field(3, ti.f32, shape=gg)
blur = ti.Vector.field(3, ti.f32, shape=gg)
final = ti.Vector.field(3, ti.f32, shape=gg)

place_p()

for frame in range(frame_T):
    update_p()

    camera.track_user_inputs(window, movement_speed=2, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((1, 1, 1))

    scene.particles(pos, radius=0.2, per_vertex_color=color)
    scene.particles(pos_trou_noir, radius=r_h * 2.6, color=(0, 0, 0))

    canvas.scene(scene)
    window.show()
    window.save_image(f"/Users/tristannormandeau/Desktop/blackHole/frame_{frame:04d}.png")