from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.0,
    'axes.titlesize': 10.2,
    'axes.labelsize': 9.2,
    'legend.fontsize': 8.2,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.2,
    'figure.dpi': 160,
    'savefig.dpi': 300,
})
COL = {
    'blue':'#1F5A8A', 'teal':'#1A7A78', 'gold':'#C78300',
    'purple':'#6F4A8E', 'red':'#B84A4A', 'gray':'#4B535B',
    'light':'#EAF2F8', 'green':'#2E7D5B'
}

def save(fig, name):
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png', bbox_inches='tight')
    plt.close(fig)

# Shared stylized aggregate
branches=[[(0,0),(.5,.2),(1.05,.55),(1.55,1.2)],[(.45,.18),(.85,-.45),(1.35,-.9)],
          [(0,0),(-.45,.35),(-1.0,.75),(-1.45,1.45)],[(-.35,.25),(-.8,-.25),(-1.35,-.45),(-1.8,-1.2)],
          [(0,0),(.05,-.55),(.2,-1.2),(.65,-1.75)]]

def draw_cluster(ax, lw=5, color='#333333'):
    for br in branches:
        x,y=zip(*br)
        ax.plot(x,y,lw=lw,solid_capstyle='round',color=color,zorder=3)

# 1 restart schematic
fig,axes=plt.subplots(1,2,figsize=(7.2,3.75),constrained_layout=True)
for ax, exact in zip(axes,[True,False]):
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-3.3,3.3); ax.set_ylim(-3.75,3.35)
    draw_cluster(ax)
    ax.add_patch(Circle((0,0),1.95,fill=False,ls='--',lw=1.2,color=COL['gray']))
    ax.add_patch(Circle((0,0),2.85,fill=False,lw=1.2,color=COL['gray']))
    t=np.linspace(0,1,90)
    x=1.30+1.50*t+0.10*np.sin(10*np.pi*t)
    y=0.55+0.25*t+0.14*np.sin(7*np.pi*t+0.4)
    ax.plot(x,y,lw=1.5,color=COL['blue'])
    ax.annotate('',xy=(2.84,y[-1]),xytext=(2.58,y[-8]),arrowprops=dict(arrowstyle='->',lw=1.4,color=COL['blue']))
    if exact:
        th=0.35; p=(1.95*np.cos(th),1.95*np.sin(th))
        ax.annotate('',xy=p,xytext=(2.73,0.75),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.25',lw=1.6,color=COL['teal']))
        ax.scatter(*p,s=30,color=COL['teal'],zorder=5)
        ax.set_title('Exact return keeps directional memory', pad=10)
        ax.text(0,-3.55,'Leaving on the right makes a return\nnear the right more likely.',ha='center',va='bottom')
    else:
        angles=np.array([0.25,1.1,2.25,3.1,4.2,5.15])
        pts=np.c_[1.95*np.cos(angles),1.95*np.sin(angles)]
        ax.scatter(pts[:,0],pts[:,1],s=20,color=COL['gold'],alpha=.68)
        chosen=pts[3]
        ax.annotate('',xy=chosen,xytext=(2.73,0.75),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-0.2',lw=1.6,color=COL['gold']))
        ax.set_title('Uniform restart erases that memory', pad=10)
        ax.text(0,-3.55,'The replacement may start anywhere,\nincluding on the opposite side.',ha='center',va='bottom')
    ax.text(-1.38,1.52,'launch circle',ha='center',va='bottom',fontsize=8,color=COL['gray'])
    ax.text(-1.88,2.27,'death circle',ha='center',va='bottom',fontsize=8,color=COL['gray'])
save(fig,'restart_schematic')

# 2 self-consistency schematic
fig,axes=plt.subplots(1,2,figsize=(7.2,3.25),constrained_layout=True)
for ax in axes:
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-3.2,3.3); ax.set_ylim(-3.0,3.1)
    draw_cluster(ax)
# trial center
ax=axes[0]
c=np.array([-0.55,-0.25]); phi=np.array([-0.05,0.25]); R=2.35
ax.add_patch(Circle(c,R,fill=False,ls='--',lw=1.2,color=COL['blue']))
ax.scatter(*c,s=55,marker='x',lw=2,color=COL['blue'],label='trial center')
ax.scatter(*phi,s=55,marker='o',facecolor=COL['gold'],edgecolor='black',lw=.5,label='mean attachment')
ax.add_patch(FancyArrowPatch(c,phi,arrowstyle='->',mutation_scale=12,lw=1.5,color=COL['red']))
ax.text(c[0]-0.1,c[1]-0.35,'c',color=COL['blue'],ha='center')
ax.text(phi[0]+0.1,phi[1]+0.25,r'$\Phi_\rho(c)$',color=COL['gold'],ha='center')
ax.set_title('A trial center produces an off-center mean')
ax.text(0,-2.85,'Move the circle toward the mean attachment point.',ha='center',va='bottom')
# fixed point
ax=axes[1]
cstar=np.array([-0.18,0.10]); R=2.25
ax.add_patch(Circle(cstar,R,fill=False,ls='--',lw=1.2,color=COL['teal']))
ax.scatter(*cstar,s=90,marker='*',facecolor=COL['teal'],edgecolor='black',lw=.5)
ax.text(cstar[0],cstar[1]+0.36,r'$c_\rho=\Phi_\rho(c_\rho)$',color=COL['teal'],ha='center')
ax.set_title('At a self-consistent center, the dipole vanishes')
ax.text(0,-2.85,'The circle center and the finite-law barycenter coincide.',ha='center',va='bottom')
save(fig,'self_center_schematic')

# 3 local error bounds
rho=np.logspace(math.log10(1.28),2,400)
generic=np.minimum(1.0,1/(rho**2-1))
selfb=np.minimum(1.0,1/(rho**2*(rho**2-1)))
fig,ax=plt.subplots(figsize=(6.7,4.2),constrained_layout=True)
ax.loglog(rho,generic,lw=2.1,color=COL['purple'],label=r'arbitrary center: $\min\{1,(\rho^2-1)^{-1}\}$')
ax.loglog(rho,selfb,lw=2.1,color=COL['teal'],label=r'self-centered: $\min\{1,[\rho^2(\rho^2-1)]^{-1}\}$')
# reference slopes normalized at rho=8
x=np.array([4,40])
ax.loglog(x, generic[np.argmin(np.abs(rho-8))]*(x/8)**-2,ls=':',lw=1.1,color=COL['purple'])
ax.loglog(x, selfb[np.argmin(np.abs(rho-8))]*(x/8)**-4,ls=':',lw=1.1,color=COL['teal'])
ax.text(25,1.7e-3,r'slope $-2$',color=COL['purple'])
ax.text(13,1.0e-6,r'slope $-4$',color=COL['teal'])
ax.set_xlabel(r'death-radius ratio $\rho=d/R$')
ax.set_ylabel('rigorous one-step TV upper bound')
ax.set_title('Self-consistency removes the leading second-order term')
ax.grid(True,which='both',alpha=.23)
ax.legend(frameon=False,loc='lower left')
save(fig,'local_bounds')

# 4 path coupling diagram
fig,ax=plt.subplots(figsize=(7.2,3.25),constrained_layout=True)
ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(-1.4,2.6)
xs=[.8,2.7,4.6,6.5,8.4]
for y,label,col in [(1.6,'exact DLA',COL['blue']),(.1,'finite-boundary DLA',COL['teal'])]:
    ax.text(.05,y,label,ha='left',va='center',weight='bold',color=col)
    for i,x in enumerate(xs):
        ax.add_patch(Circle((x,y),.29,facecolor='white',edgecolor=col,lw=1.5))
        ax.text(x,y,str(i),ha='center',va='center',fontsize=8,color=col)
        if i<len(xs)-1:
            ax.annotate('',xy=(xs[i+1]-.34,y),xytext=(x+.34,y),arrowprops=dict(arrowstyle='->',lw=1.3,color=col))
for x in xs[:-1]:
    ax.plot([x,x],[.39,1.31],ls='--',lw=1,color=COL['gray'],alpha=.7)
ax.text(4.6,2.28,'Maximally couple the next attachment only while the histories agree',ha='center',va='center')
ax.annotate('first mismatch',xy=(6.5,.78),xytext=(7.8,-.95),ha='center',color=COL['red'],arrowprops=dict(arrowstyle='->',color=COL['red'],lw=1.2))
ax.text(4.55,-1.15,r'probability of no mismatch through $N$: $\prod_{n<N}(1-\epsilon_n)$',ha='center',va='center',color=COL['gray'])
save(fig,'path_coupling')

# 5 global radius requirement
M=np.logspace(1,8,300)
delta=0.01
ebar=1-(1-delta)**(1/M)
x=(1+np.sqrt(1+4/ebar))/2
selfrho=np.sqrt(x)
genrho=np.sqrt(1+1/ebar)
fig,ax=plt.subplots(figsize=(6.7,4.2),constrained_layout=True)
ax.loglog(M,selfrho,lw=2.1,color=COL['teal'],label='self-centered fourth-order certificate')
ax.loglog(M,genrho,lw=2.1,color=COL['purple'],label='generic second-order certificate')
ax.set_xlabel('number of growth steps in the certified horizon')
ax.set_ylabel(r'sufficient constant ratio $\rho$')
ax.set_title('Worst-case boundary growth for a 1% path-law budget')
ax.grid(True,which='both',alpha=.23)
ax.legend(frameon=False)
ax.text(1.4e5,130,r'$\rho\asymp M^{1/4}$',color=COL['teal'])
ax.text(2e4,2e4,r'$\rho\asymp M^{1/2}$',color=COL['purple'])
save(fig,'global_radius_rule')

# 6 calibration cost
rho=np.logspace(math.log10(1.5),2,240)
eta=1e-3
Mreq=4*rho**4*np.log(4/eta)
fig,ax=plt.subplots(figsize=(6.7,4.2),constrained_layout=True)
ax.loglog(rho,Mreq,lw=2.1,color=COL['red'])
ax.set_xlabel(r'death-radius ratio $\rho$')
ax.set_ylabel('probe count needed for sampling error of fourth-order size')
ax.set_title('A rigorous Monte Carlo certificate is expensive if recalibrated every step')
ax.grid(True,which='both',alpha=.23)
ax.text(8,2.2e7,r'$M_{\rm probe}\gtrsim 4\rho^4\log(4/\eta)$',color=COL['red'])
save(fig,'calibration_cost')

# 7 amortized block schedule
fig, ax = plt.subplots(figsize=(7.2, 3.45), constrained_layout=True)
ax.set_xlim(0, 12.2); ax.set_ylim(-0.35, 3.15); ax.axis('off')
starts = np.array([0.7, 3.1, 6.0, 9.45])
lengths = np.array([2.0, 2.45, 2.95, 2.15])
for i, (x0, Lb) in enumerate(zip(starts, lengths)):
    y = 1.18
    ax.plot([x0, x0+Lb], [y, y], lw=6, solid_capstyle='butt', color=COL['light'])
    ax.plot([x0, x0+Lb], [y, y], lw=1.15, color=COL['blue'])
    ax.scatter([x0], [y], s=62, marker='o', facecolor=COL['gold'], edgecolor='black', lw=.5, zorder=5)
    ax.text(x0, y-0.34, rf'$\tau_{i}$', ha='center', va='top', fontsize=8)
    ax.text(x0+Lb/2, y-0.11, 'reuse one center', ha='center', va='top', fontsize=8, color=COL['gray'])
    # drift wedge
    xs=np.linspace(x0, x0+Lb, 40)
    upper=1.62+0.13*(xs-x0)/Lb
    lower=1.62-0.13*(xs-x0)/Lb
    ax.fill_between(xs,lower,upper,alpha=.20,color=COL['teal'])
    ax.plot(xs,1.62+0.05*np.sin(np.linspace(0,2.2,len(xs))),lw=1.2,color=COL['teal'])
    ax.text(x0+Lb/2,1.92,'controlled center drift',ha='center',fontsize=8,color=COL['teal'])
    if i < len(starts)-1:
        ax.annotate('',xy=(starts[i+1]-.08,y),xytext=(x0+Lb+.08,y),
                    arrowprops=dict(arrowstyle='->',lw=1.1,color=COL['gray']))
# probe batch depiction
ax.text(.25,2.86,'At each block boundary:',weight='bold',ha='left',va='center')
ax.scatter(np.linspace(.55,1.45,7),np.full(7,2.53),s=22,facecolor=COL['gold'],edgecolor='black',lw=.3)
ax.text(1.70,2.53,'probe attachments',va='center')
ax.annotate('',xy=(4.28,2.53),xytext=(3.72,2.53),arrowprops=dict(arrowstyle='->',lw=1.2,color=COL['gray']))
ax.scatter([4.62],[2.53],s=75,marker='*',facecolor=COL['teal'],edgecolor='black',lw=.4)
ax.text(4.96,2.53,'empirical mean becomes the next center',va='center')
ax.text(6.05,.16,r'block length $H_k\asymp \tau_k^{1/4-\gamma}$ grows with the cluster',
        ha='center',va='center',color=COL['gray'])
ax.set_title('Amortization: calibrate rarely, control staleness between calibrations',pad=8)
save(fig,'amortized_schedule')

# 8 feasible exponent region
fig, ax = plt.subplots(figsize=(6.7, 4.5), constrained_layout=True)
a = np.linspace(.25,.60,500)
lower = np.maximum(0,1-2*a)
upper = np.full_like(a,1/12)
mask = lower < upper
ax.fill_between(a[mask],lower[mask],upper[mask],alpha=.28,color=COL['teal'],label='proved sublinear-overhead region')
ax.plot(a,1-2*a,lw=1.8,color=COL['purple'],label=r'summability boundary $\gamma=1-2\alpha$')
ax.axhline(1/12,lw=1.8,ls='--',color=COL['red'],label=r'probe-work boundary $\gamma=1/12$')
ax.axvline(11/24,lw=1.25,ls=':',color=COL['gold'])
ax.axvline(.25,lw=1.0,ls=':',color=COL['gray'])
ax.axvline(.5,lw=1.0,ls=':',color=COL['gray'])
ax.text(11/24+.006,.105,r'$11/24$',color=COL['gold'])
ax.text(.252,.006,'ideal exact-self-center\nthreshold $1/4$',fontsize=8,color=COL['gray'])
ax.text(.502,.006,'generic worst-case\nthreshold $1/2$',fontsize=8,color=COL['gray'])
ax.set_xlim(.25,.60); ax.set_ylim(0,.13)
ax.set_xlabel(r'boundary exponent $\alpha$ in $\varrho_n=L n^\alpha$')
ax.set_ylabel(r'calibration accuracy exponent $\gamma$')
ax.set_title('Where summable staleness and sublinear probe work coexist')
ax.grid(alpha=.18)
ax.legend(frameon=False,loc='upper right')
save(fig,'amortized_feasible_region')
