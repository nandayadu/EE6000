------------------------------------------------------
#IMPORTS
------------------------------------------------------

import torch
import matplotlib.pyplot as plt
import matplotlib
import PIL
import numpy as np
import cv2
import random
from scipy.special import comb, factorial


------------------------------------------------------------
#USER INPUTS
------------------------------------------------------------

lambda_m=550e-9#632.8e-9#
size_x_in_m = 150e-7#4*16.384e-6#
size_y_in_m = 150e-7#4*16.384e-6#
period_x=600e-9#768e-9#
period_y=np.Inf#768e-9#
order_num_x=2
order_num_y=0
screen_size_x_in_m = 1.5e-4#204.8e-6#
screen_size_y_in_m = 1.5e-4#204.8e-6#

# screen resolution improvement factor (by default 128 x 128)
# simulation time ~ 4+(8*improve_fact_x*improve_fact_y) sec for 512 x 152 grating
improve_fact_x =1
improve_fact_y =1

R=0.1e-2; # distance of screen from gating 

Nx = int(512); # # of pixels in x-dimension 
Ny = int(512);  # # of pixels in y-dimension

---------------------------------------------------------------------
#INCIDENT BEAM
------------------------------------------------------------------------

l=0
p=0
w0=40e-6
z = 0.0
Dcenter_x_in_m=0.0
Dcenter_y_in_m=0.0
input_intensity = 1.0
k0=-np.complex(0,(2*np.pi)/lambda_m)

if w0 is None:
  w0 = size_x_in_m/2;               # Beam waist

zR = k0*w0**2.0/2;       # Calculate the Rayleigh range

ci=np.sqrt(2*factorial(p)/(np.pi*factorial(p+l)))
# Setup the cartesian grid for the plot at plane z
xx, yy = np.zeros([Ny,Nx,3]), np.zeros([Ny,Nx,3])
xx, yy = np.meshgrid(np.linspace(-size_x_in_m/2-Dcenter_x_in_m, size_x_in_m/2-Dcenter_x_in_m,Nx), np.linspace(-size_y_in_m/2-Dcenter_y_in_m, size_y_in_m/2-Dcenter_y_in_m,Ny));

# Calculate the cylindrical coordinates
r = np.sqrt(xx**2 + yy**2);
phi = np.arctan2(yy, xx);

U00 = 1.0/(1 + z/zR) * np.exp(-r**2.0/w0**2/(1 + z/zR));

#w = w0 * np.sqrt(1.0 + z**2/zR**2);
w= w0 * np.abs(1 + z/zR)
Ri = np.sqrt(2.0)*r/w;

# Lpl from OT toolbox (Nieminen et al., 2004)
Lpl = comb(p+l,p) * np.ones(np.shape(Ri));   # x = R(r, z).^2
for m in range(1, p+1):
  Lpl = Lpl + (-1.0)**m/factorial(m) * comb(p+l,p-m) * Ri**(2.0*m);

U = ci*U00*Ri**l*Lpl*np.exp(1j*l*phi)*np.exp(-1j*(2*p + l + 1)*np.arctan(z/np.abs(zR)));
incident_LGbeam_intensityNphase = np.abs(np.sqrt(input_intensity/np.double(Nx*Ny))*U);
#print(incident_LGbeam_intensityNphase)
#plt.imshow(incident_LGbeam_intensityNphase);
fig = plt.figure(figsize=(15, 15)) # set the height and width in inches
    
# plot input beam Intensity profile
plt.subplot(1,3,1)
#incident_LGbeam_intensityNphase[0,0]=0
plt.imshow(incident_LGbeam_intensityNphase, cmap='hot')#,cmap='copper', interpolation='nearest')
print(incident_LGbeam_intensityNphase)
plt.colorbar(fraction=0.045)

----------------------------------------------------------------------------------------------------
#ANGULAR ILLUMINATION OF INCIDENT BEAM ON TO THE GRATING
----------------------------------------------------------------------------------------------------

incidence_theta_x=np.arcsin(order_num_x*lambda_m/period_x)
incidence_theta_y=np.arcsin(order_num_y*lambda_m/period_y)
phasex=(2*np.pi*np.sin(incidence_theta_x)/lambda_m)*np.linspace(-size_x_in_m/2,size_x_in_m/2,Nx)
phasey=(2*np.pi*np.sin(incidence_theta_y)/lambda_m)*np.linspace(-size_y_in_m/2,size_y_in_m/2,Ny)
phasexx, phaseyy = np.meshgrid(phasex, phasey, sparse=True)
C_phase= phasexx+phaseyy
print(C_phase.shape)
print(incidence_theta_x)
print(incidence_theta_y)

fig = plt.figure(figsize=(15, 15)) # set the height and width in inches
plt.subplot(1,3,1)
plt.imshow(C_phase)#,cmap='jet')
plt.axis('off')
plt.show()

----------------------------------------------------------------------------------------------------------
#CIRCULAR APERTURE GRATING
----------------------------------------------------------------------------------------------------------


ell=1; # This is the value of the topological charge of the LG beam

s=2*np.pi/Nx*size_x_in_m/period_x; # This parameter defines the fringe density. Use s=1.
C1 = np.zeros([Ny,Nx]); # matrix that will become the image 

for x in range(Nx):
    for y in range(Ny):
        x0 = Nx/2; # coordinates of the center of the image
        y0 = Ny/2;
        xr=x-x0;
        yr=y-y0;
        r=np.sqrt(xr**2+yr**2); # radial coordinate
        #if((xr<=Nx/2) or (yr<=Ny/2)):
        if r<=Nx/2:#/aperture:
          phi=np.arctan2(yr,xr);  # angular coordinate
          phi2 = ell*phi                              #------1D grating if ell=0, else Fork grating
          phi1 = phi2 + s*(xr); #phase of fork1 at pixel (x,y);
          r1 = np.mod(phi1,2*np.pi)/(2*np.pi); #phase mod 2 pi in units of 2pi
          if r1 >=0.5:
            r1=1;
          else:
            r1=0;
          C1[y,x] = r1; 



matplotlib.image.imsave('grating1.jpg', C1)

fig = plt.figure(figsize=(10, 10))
img = cv2.imread('grating1.jpg', 2)
ret, C1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
C1 = C1/255
plt.subplot(1,3,2)
plt.imshow(C1, cmap='gray')


-------------------------------------------------------------------------------------------------------------
#SQUARE APERTURE
-------------------------------------------------------------------------------------------------------------
ell=1; # This is the value of the topological charge of the LG beam

s=2*np.pi/Nx*size_x_in_m/period_x; # This parameter defines the fringe density. Use s=1.
C2 = np.zeros([Ny,Nx]); # matrix that will become the image 

for x in range(Nx):
    for y in range(Ny):
        x0 = Nx/2; # coordinates of the center of the image
        y0 = Ny/2;
        xr=x-x0;
        yr=y-y0;
        r=np.sqrt(xr**2+yr**2); # radial coordinate
        #if((xr<=Nx/2) or (yr<=Ny/2)):
        if((x>=Nx/16) and (y>=Nx/16) and (x<=15*Nx/16) and (y<=15*Ny/16)):
          phi=np.arctan2(yr,xr);  # angular coordinate
          phi2 = ell*phi                              #------1D grating if ell=0, else Fork grating
          phi1 = phi2 + s*(xr); #phase of fork1 at pixel (x,y);
          r1 = np.mod(phi1,2*np.pi)/(2*np.pi); #phase mod 2 pi in units of 2pi
          if r1 >=0.5:
            r1=1;
          else:
            r1=0;
          C2[y,x] = r1; 



matplotlib.image.imsave('grating1.jpg', C2)

fig = plt.figure(figsize=(10, 10))
img = cv2.imread('grating1.jpg', 2)
ret, C2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
C2 = C2/255
plt.subplot(1,3,2)
plt.imshow(C2, cmap='gray')

------------------------------------------------------------------------------------------------------------
#TRIANGULAR APERTURE
------------------------------------------------------------------------------------------------------------

ell=1; # This is the value of the topological charge of the LG beam

s=2*np.pi/Nx*size_x_in_m/period_x; # This parameter defines the fringe density. Use s=1.
C3 = np.zeros([Ny,Nx]); # matrix that will become the image 

for x in range(Nx):
    for y in range(Ny):
        x0 = Nx/2; # coordinates of the center of the image
        y0 = Ny/2;
        xr=x-x0;
        yr=y-y0;
        r=np.sqrt(xr**2+yr**2); # radial coordinate
        #if((xr<=Nx/2) or (yr<=Ny/2)):
        if r<=Nx:#/aperture:
          phi=np.arctan2(yr,xr);  # angular coordinate
          phi2 = ell*phi                              #------1D grating if ell=0, else Fork grating
          phi1 = phi2 + s*(xr); #phase of fork1 at pixel (x,y);
          r1 = np.mod(phi1,2*np.pi)/(2*np.pi); #phase mod 2 pi in units of 2pi
          if r1 >=0.5:
            r1=1;
          else:
            r1=0;
          C3[y,x] = r1; 
          
          
Nx_new=600
Ny_new=600
Angle=-27
C4=np.ones([Nx_new,Ny_new])
C41=np.ones([Nx_new,Ny_new])
C42=np.ones([Nx_new,Ny_new])
#plt.imshow(C_new)
C_new=cv2.getRotationMatrix2D(center=(0, Ny_new), angle=Angle, scale=1)
C_new = cv2.warpAffine(C4, C_new, (Nx_new, Ny_new))
C_new=C_new[88:600,0:512]
# cv2.rotate(C_new, cv2.ROTATE_60_CLOCKWISE)
Angle1=27
C_new1=cv2.getRotationMatrix2D(center=(Nx_new, Ny_new), angle=Angle1, scale=1)
C_new1 = cv2.warpAffine(C41, C_new1, (Nx_new, Ny_new))
C_new1=C_new1[88:600,88:600]

matplotlib.image.imsave('gratingn.jpg', C_new)
fig = plt.figure(figsize=(10, 10))
img = cv2.imread('gratingn.jpg', 2)
ret, C4 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
C4 = C4/255
# plt.subplot(2,3,1)
# plt.imshow(C4, cmap='gray')
# plt.colorbar(fraction=0.045);

matplotlib.image.imsave('gratingn.jpg', C_new1)
fig = plt.figure(figsize=(10, 10))
img = cv2.imread('gratingn.jpg', 2)
ret, C41 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
C41 = C41/255
# plt.subplot(2,3,2)
# plt.imshow(C41, cmap='gray')
# plt.colorbar(fraction=0.045);

C_new2=C4+C41
matplotlib.image.imsave('gratingn.jpg', C_new2)
fig = plt.figure(figsize=(10, 10))
img = cv2.imread('gratingn.jpg', 2)
ret, C42 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
C42 = C42/255
# plt.subplot(2,3,3)
# plt.imshow(C42, cmap='gray')
# plt.colorbar(fraction=0.045);


C_tri= C42+C3
matplotlib.image.imsave('gratingn.jpg', C_tri)
fig = plt.figure(figsize=(10, 10))
img = cv2.imread('gratingn.jpg', 2)
ret, C_tri = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
C_tri = C_tri/255
plt.subplot(2,3,3)
plt.imshow(C_tri, cmap='gray')
plt.colorbar(fraction=0.045);



---------------------------------------------------------------------------------------------------------------
#APERTURE SELECTION FOR AMPLITUDE GRATING
---------------------------------------------------------------------------------------------------------------

C= C1; # For Circular
# C= C2; # For Square
# C= C_tri; # For Triangular

#grating_type='amplitude grating'
C_n=C*incident_LGbeam_intensityNphase # grating and amplitude multiplied togather to create a single matrix for amplitude modulation

matplotlib.image.imsave('grating5.jpg', C_n)
fig = plt.figure(figsize=(10, 10))
img = cv2.imread('grating5.jpg', 2)
ret, C_n = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
C_n = C_n/255
plt.subplot(1,3,1)
plt.imshow(C_n, cmap='gray')
plt.colorbar(fraction=0.045);

plt.subplot(1,3,3)
plt.imshow(C_phase)
plt.colorbar(fraction=0.045);


----------------------------------------------------------------------------------------------------------------
#DIFFRACTION ORDER
----------------------------------------------------------------------------------------------------------------

k=-np.complex(0,(2*np.pi*n3)/lambda_m) # wavelength of light in medium n=1.5
R2=R**2

order_xpos_in_m=np.tan(np.arcsin(lambda_m*order_num_x/period_x)-incidence_theta_x)*R
order_ypos_in_m=np.tan(np.arcsin(lambda_m*order_num_y/period_y)-incidence_theta_y)*R
print(order_xpos_in_m)
print(order_ypos_in_m)


-----------------------------------------------------------------------------------------------------------------
# VECTOR MESH FOR GRATING AND SCREEN
-----------------------------------------------------------------------------------------------------------------

#splitting factor
g_division_fact_x =int(C_n.shape[0]/128)
g_division_fact_y =int(C_n.shape[1]/128)


#splitting the grating
grating_split_x=np.linspace(-0.5,0.5,1+g_division_fact_x)*size_x_in_m
grating_split_y=np.linspace(-0.5,0.5,1+g_division_fact_y)*size_y_in_m
print(grating_split_x)

# grating_split_x=np.linspace(-0.5,0.5,1+improve_fact_x)*size_x_in_m
# grating_split_y=np.linspace(-0.5,0.5,1+improve_fact_y)*size_y_in_m


#splitting the grating pixels
grating_p_split_x=(np.linspace(0,1,1+g_division_fact_x)*C_n.shape[0]).astype('int')
grating_p_split_y=(np.linspace(0,1,1+g_division_fact_y)*C_n.shape[1]).astype('int')
print(grating_p_split_x)
print(grating_p_split_y)

# grating_p_split_x=(np.linspace(0,1,1+improve_fact_x)*C.shape[0]).astype('int')
# grating_p_split_y=(np.linspace(0,1,1+improve_fact_y)*C.shape[1]).astype('int')

screen_size_x_in_pixel = 128
screen_size_y_in_pixel = 128

#splitting the screen
screen_split_x=np.linspace(-0.5,0.5,1+improve_fact_x)*screen_size_x_in_m
screen_split_y=np.linspace(-0.5,0.5,1+improve_fact_y)*screen_size_y_in_m
print(screen_split_x)

-----------------------------------------------------------------------------------------------------------------
#HUYGEN'S PRINCIPLE CONSTRUCT
-----------------------------------------------------------------------------------------------------------------

pf= torch.zeros(screen_size_x_in_pixel*improve_fact_x,0).to(cuda)
for r_unit in range(improve_fact_x):
  #print(r_unit)
  S_x_vec=torch.linspace(order_xpos_in_m+screen_split_x[r_unit],order_xpos_in_m+screen_split_x[r_unit+1],screen_size_x_in_pixel).to(cuda); 
  pf_y= torch.zeros((0,screen_size_y_in_pixel)).to(cuda)
  #print(S_x_vec)
  for c_unit in range(improve_fact_y): 
    #print(c_unit)  
    S_y_vec=torch.linspace(order_ypos_in_m+screen_split_y[c_unit],order_ypos_in_m+screen_split_y[c_unit+1],screen_size_y_in_pixel).to(cuda);
    S_vec=torch.zeros((S_x_vec.size()[0],S_y_vec.size()[0])).to(cuda)
    #print(S_y_vec)
    S_vec=S_y_vec.view(S_y_vec.size()[0],1)*1j + S_vec
    S_vec=(S_x_vec + S_vec)
    #print(S_vec)
    #print(S_vec.size())
    #S_vec.view(S_x_vec.size()[0]*S_y_vec.size()[0],)

    fp_unit= torch.zeros(  S_x_vec.size()[0]*S_y_vec.size()[0],1).to(cuda)
    #print(fp_unit)
    for gr_unit in range(g_division_fact_x):
      #print(gr_unit)
      G_x_vec=torch.linspace(grating_split_x[gr_unit],grating_split_x[gr_unit+1],int(C_n.shape[0]/g_division_fact_x)).to(cuda)
      #print(G_x_vec)
      for gc_unit in range(g_division_fact_y):
        G_y_vec=torch.linspace(grating_split_y[gc_unit],grating_split_y[gc_unit+1],int(C_n.shape[1]/g_division_fact_y)).to(cuda)
        G_vec=torch.zeros((G_x_vec.size()[0],G_y_vec.size()[0])).to(cuda)

        G_vec=G_y_vec.view(G_y_vec.size()[0],1)*1j + G_vec
        G_vec=(G_x_vec + G_vec)
        #print(G_vec)
        #G_vec.view(1,G_x_vec.size()[0]*G_y_vec.size()[0])

        #E0=torch.complex(torch.tensor(C[grating_p_split_y[gc_unit]:grating_p_split_y[gc_unit+1],grating_p_split_x[gr_unit]:grating_p_split_x[gr_unit+1],1],dtype=torch.float),torch.zeros(int(C.shape[0]/g_division_fact_x),int(C.shape[1]/g_division_fact_y))).to(cuda)
        E0=torch.polar(torch.tensor(C_n[grating_p_split_y[gc_unit]:grating_p_split_y[gc_unit+1],grating_p_split_x[gr_unit]:grating_p_split_x[gr_unit+1]],dtype=torch.float,device=cuda),torch.tensor(C_phase[grating_p_split_y[gc_unit]:grating_p_split_y[gc_unit+1],grating_p_split_x[gr_unit]:grating_p_split_x[gr_unit+1]],dtype=torch.float,device=cuda))
        #print(E0)
        #Calculate optical path from each point on the grating to each point on the screen:
        path_diff=torch.zeros((S_x_vec.size()[0]*S_y_vec.size()[0],G_x_vec.size()[0]*G_y_vec.size()[0]),device=cuda)
        path_diff=path_diff+G_vec.view(1,G_x_vec.size()[0]*G_y_vec.size()[0])
        path_diff=path_diff-S_vec.view(S_x_vec.size()[0]*S_y_vec.size()[0],1)
        path_diff=torch.exp(torch.sqrt(path_diff*path_diff.conj()+R2)*k)#/path_diff
        
        #Calculate the diffraction pattern:
        #E0=torch.complex(torch.ones((G_x_vec.size()[0],G_y_vec.size()[0])),torch.zeros((G_x_vec.size()[0],G_y_vec.size()[0])))
        E0=E0.reshape(G_x_vec.size()[0]*G_y_vec.size()[0],1)
        #print(E0)
        fp_unit=fp_unit+torch.mm(path_diff,E0)
        #fp_unit=fp_unit-1j*R*torch.mm(path_diff,E0)/lambda_m
    #plt.imshow(np.abs(torch.abs(E0).view(G_x_vec.size()[0],G_y_vec.size()[0])).numpy())
    pf_y= torch.cat((  pf_y,fp_unit.view(S_x_vec.size()[0],S_y_vec.size()[0])    ),0)
  pf= torch.cat((  pf,pf_y   ),1)
print(pf.shape) 

-----------------------------------------------------------------------------------------------------------------
#INTENSITY VORTEX AT SCREEN 
-----------------------------------------------------------------------------------------------------------------

plt.imshow(np.abs(pf.to(cpu)),cmap='hot')
plt.axis('off')
#plt.colorbar();
matplotlib.image.imsave('magnitude.jpg', np.abs(pf.to(cpu)).numpy(),cmap='gray')




