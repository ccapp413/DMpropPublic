#!/usr/bin/python3
import os
import os.path
import csv
import numpy as np
import math
import scipy.special
from scipy.interpolate import interp1d
from scipy.interpolate import splev
from scipy.interpolate import splrep
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import trapz
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.signal import convolve

#a numerical factor that speeds up energy convolution
#setting this to even smaller values may speed up the integration, but has NOT been tested for values below 10**-10
speedfactor=10**-9

#define constants and functions
#the incomplete gamma function with first argument zero is related to the exponential integral as below
amu = 0.932

def gammazero(x):
    return -scipy.special.expi(-x)#*np.heaviside(20-x,0.5)

def mu(mdm,A):
    if A==1:
        return mdm*0.938/(mdm+0.938)
    else:
        return mdm*A*amu/(mdm + A*amu)

def sigmaA(mdm,logsigma,A):
    return 10.0**logsigma*(A**2)*(mu(mdm,A)/mu(mdm,1))**2

rho_earth = 2.7
#from 1802.04764
n_A = 6.022*10.0**23.0
f_O = 0.466#/.985
f_Si = 0.277#/.985
f_Al = 0.081#/.985
f_Fe = 0.050#/.985
f_Ca = 0.036#/.985
f_K = 0.028#/.985
f_Na = 0.026#/.985
f_Mg = 0.021#/.985

rhoatm=1
f_O_atm=0.21*16/(.21*16+.79*14)
f_N_atm=0.79*14/(.21*16+.79*14)

#make directories to hold output files
Path("depth").mkdir(exist_ok=True)
Path("energy").mkdir(exist_ok=True)

#input parameters
shield = int(input("Please type 1 for crust shielding, or 0 for atmospheric (default is crust): ") or 1)
mdm = float(input("Please input dark matter mass in GeV (default is 1.0): ") or 1.0)
logsigma = float(input("Please input Log_10 of the DM-nucleon cross section in units of cm^2 (Log_10[sigma/cm^2]; default is -30): ") or -30.0)
depth=1
if shield==1:
    depth = float(input("Please input detector depth in m (default is 1000): ") or 1000)
    if depth < 10:
        print("Warning: you have chosen crust shielding, but have set the detector depth to less than 10 m. Note that this code will not account for atmospheric attenuation, which may become significant for depth << 10 m.")
itermax = int(input("Please choose the number of convolutions to perform (default is zero): ") or 0)+1
if itermax == 0:
    print("Warning: zero iterations selected. Outputting only the distribution of first scattering depths and fraction of particles reaching the detector without scattering.")

#compute mean free path
linv_O = (sigmaA(mdm,logsigma,16.0) * f_O * rho_earth * n_A / 16.0)
linv_Si = (sigmaA(mdm,logsigma,28.0) * f_Si * rho_earth * n_A / 28.0)
linv_Al = (sigmaA(mdm,logsigma,27.0) * f_Al * rho_earth * n_A / 27.0)
linv_Fe = (sigmaA(mdm,logsigma,56.0) * f_Fe * rho_earth * n_A / 56.0)
linv_Ca = (sigmaA(mdm,logsigma,40.0) * f_Ca * rho_earth * n_A / 40.0)
linv_K = (sigmaA(mdm,logsigma,39.0) * f_K * rho_earth * n_A / 39.0)
linv_Na = (sigmaA(mdm,logsigma,23.0) * f_Na * rho_earth * n_A / 23.0)
linv_Mg = (sigmaA(mdm,logsigma,24.0) * f_Mg * rho_earth * n_A / 24.0)

linv_O_atm = sigmaA(mdm,logsigma,16)*f_O_atm*rhoatm*n_A/16.0
linv_N_atm = sigmaA(mdm,logsigma,14)*f_N_atm*rhoatm*n_A/14.0

nucleuslist=[16,28,27,56,40,39,23,24]
linvlist=[linv_O,linv_Si,linv_Al,linv_Fe,linv_Ca,linv_K,linv_Na,linv_Mg]
#print("linvlist")
#print(linvlist)

res=1000
l=(1.0/(linv_O+linv_Si+linv_Al+linv_Fe+linv_Ca+linv_K+linv_Na+linv_Mg))/(100.0)*(res/depth)
print(l)

if shield==0:
    l=(1.0/(linv_N_atm+linv_O_atm))/100.0*(1000./10.)
    nucleuslist=[14,16]
    linvlist=[linv_N_atm,linv_O_atm]

#distribution of first scatterings
dist=[]
dist1=[]
fractionlist=[]

#this variable tells the code whether to track particles that have backscattered from below detector depth
backscatter=0
if shield==0:
    backscatter=0

#set the distribution of first scattering depths, which for a flux that is isotropic from above is  given by an incomplete gamma function
if backscatter==0:
    xinitial=np.logspace(-10,math.log10(res),20000)
if backscatter==1:
    xinitial=np.logspace(-10,2*math.log10(res),40000)
for i in xinitial:
    dist1.append(1.0/l * gammazero(i/l))
f = interp1d(xinitial,dist1,kind='quadratic',fill_value='extrapolate')

testing=convolve(dist1,dist1)/(2*l)
disttest=[]
distbelow=[]
distconv=[]
testx=np.linspace(0,res,200000)
belowx=np.linspace(res+res/200000,2*res,400000)
testconv=np.linspace(0,2*(res),399999)
for i in testx:
    disttest.append(1.0/l * gammazero(10**(-5)+i/l)/200)
for i in belowx:
    distbelow.append(1.0/l * gammazero((i)/l)/200)
for i in testconv:
    distconv.append(1.0/(2*l) * gammazero(10**(-5)+abs((res)-i)/l)/200)
finalx=testx

print(xinitial)
print(1000+10*l)
#fraction of particles that reach at least detector depth without scattering
nd = integrate.quad(lambda x: 1.0/l * gammazero(x/l),res,res+int(10*l))[0]
print("Fraction of particles reaching detector without scattering is "+str(nd))
dist.append(dist1)
fractionlist.append(nd)

#loop over additional scatterings
natm=0.0
for iter in range(1,itermax):
    if sum(fractionlist)>0.9:
        print("More than 90 percent of DM has reached detector after "+str(iter)+" scatterings. Terminating loop.")
        break
    dist2=convolve(disttest,distconv,method="fft").tolist()
    dist3=convolve(distbelow,distconv,method="fft").tolist()

    #compute the number of particles that traveled from above the detector to below it,
    #and the fraction of particles that backscattered from below he detector to above it
    #combine the distribution of particles that scattered from above the detector and from below the detector
    #into one total distribution
    print(len(dist2))
    print(len([i for i in range(int(20*l)+res)]))

    if backscatter==0:
        nd=sum(dist2[-200000:])
    if backscatter==1:
        nd=sum(dist2[-200000:])+sum(dist3[:200000])
    if backscatter==0:
        disttest=dist2[200000:400000]
        distbelow=dist2[400000:]
    if backscatter==1:
        disttest=[sum(x) for x in zip(dist2[200000:400000],dist3[:200000])]
        distbelow=[sum(x) for x in zip(dist2[400000:],dist3[200000:400000])]
    remaining=integrate.quad(lambda x: f(x),0,res)[0]
    print("Fraction of particles reaching detector after "+str(iter)+" scatterings is "+str(nd))
    dist.append(disttest)
    fractionlist.append(nd)

##########################################################################################
#energy loss calculation
def elossmaxfrac(mdm,A):
    return 4.0*mu(mdm,A)/(mdm+A*amu)

#velocity distribution in reference frame of the Sun, from Emken et al.
def velocitydist(v):
    vrms=270./300000.
    vsun=220./300000.
    vesc=544./300000.
    vearth=240./300000.
    Nesc = math.pi * vsun * vsun * (math.sqrt(math.pi) * vsun * math.erf(vesc / vsun) - 2 * vesc * math.exp(-vesc * vesc / vsun / vsun))
    return math.pi*v*vsun*vsun/(vearth*Nesc) * (2*math.exp(-(v*v+vearth*vearth)/(vsun*vsun))*math.sinh(2*v*vearth/(vsun*vsun)) + (math.exp(-(v+vearth)**2/(vsun**2)) - math.exp(-vesc*vesc/(vsun*vsun)))*np.heaviside(np.abs(v+vearth)-vesc,0.5) - (math.exp(-(v-vearth)**2/(vsun**2)) - math.exp(-vesc*vesc/(vsun*vsun)))*np.heaviside(np.abs(v-vearth)-vesc,0.5) )*np.heaviside(vearth+vesc-v,0.5)

def edist(mdm,edm):
    return velocitydist((2*edm/mdm)**(0.5))*1.0/(2.0*mdm*edm)**0.5
energyplots=[]
energydist=[]
energylist=[]
for i in np.logspace(np.log10(.00001),np.log10(10000*mdm),10000):
    energydist.append(edist(mdm,i*10**(-9)))
    energylist.append(i*10**(-9))
fen = interp1d(energylist,energydist,kind="quadratic",bounds_error=False,fill_value=0)
energyplots.append(energydist)

#energy distributions after "itermax" scatterings
for i in range(1,len(fractionlist)):
    print(i)
    energydist2=[]
    enfilename="energy/shield"+str(shield)+"mdm"+str(mdm)+"depth"+str(depth)+"iter"+str(i)+"energy.dat"
    if Path(enfilename).exists():
        with open(enfilename, "r") as infile:
            for line in infile.readlines():
                energydist2.append(float(line))
    else:
        for edm in energylist:
            energydist2.append((1./speedfactor)*sum([linvlist[j]*integrate.romberg(lambda xp: (1.0/(1.0-xp))*(1.0/elossmaxfrac(mdm,nucleuslist[j]))*speedfactor*fen(edm/(1.0-xp)),0,elossmaxfrac(mdm,nucleuslist[j])) for j in range(len(nucleuslist))])/sum(linvlist))
        with open(enfilename, "w") as outfile:
            for element in energydist2:
                outfile.write(str(element)+"\n")
    fen = interp1d(energylist,energydist2,bounds_error=False,fill_value=0)
    energyplots.append(energydist2)

print("testing energy summation")

#multiply the energy distribution after i scatterings by the fraction of particles crossing detector depth after i scatterings
#this gives the total energy distribution at the detector
totaledist=[]
differentialedist=[]
for i in range(len(fractionlist)):
    differentialedist.append([el*fractionlist[i] for el in energyplots[i]])
    if i == 0:
        totaledist.append([el*fractionlist[0] for el in energyplots[0]])
    else:
        totaledist.append([x + y for x, y in zip(totaledist[i-1], [el*fractionlist[i] for el in energyplots[i]])])

finalvelocitydist=[]
differentialvelocitydist=[]
for j in range(len(fractionlist)):
    print(j)
    fetov = interp1d(energylist,totaledist[j],bounds_error=False,fill_value=0)
    differentialfetov = interp1d(energylist,differentialedist[j],bounds_error=False,fill_value=0)
    finalvelocitydist.append([])
    differentialvelocitydist.append([])
    for i in range(1,3000):
        finalvelocitydist[j].append(fetov(0.5*mdm*(i*10**(-6))**2)*mdm*(i*10**(-6)))
        differentialvelocitydist[j].append(differentialfetov(0.5*mdm*(i*10**(-6))**2)*mdm*(i*10**(-6)))
print("integral of initial dist is")
initialabovethreshold=sum(finalvelocitydist[0][1500:3000])/fractionlist[0]/10**6
print(initialabovethreshold)
print("fraction of particles remaining above threshold is")
print(sum(finalvelocitydist[len(fractionlist)-1][1500:3000])/initialabovethreshold/10**6)

#plot stuff
velplot=[]
eplot=[]
for i in range(1,3000):
    velplot.append(velocitydist(i/1000000))
    eplot.append(edist(mdm,i/1000000000.))

nA=6.022*10**23
exposurexenon=1042*1000*nA*34.2*(60*60*24)

fveldist=interp1d([10**-6*i for i in range(1,3000)],finalvelocitydist[len(fractionlist)-1],bounds_error=False,fill_value=0)
ereclist=np.linspace(1*10**-6,50*10**-6,1000)
espectlist=[]
for erec in ereclist:
    vmin=(erec*(mdm+131*amu)**2/(2*131*amu*mdm**2))**(0.5)
    vmax=0.003
    espectlist.append((3.0*10**10/(1.0))*(exposurexenon/(131*amu))*(0.5*0.3/mdm)*(mdm+131*amu)**2/(2*131*amu*mdm**2)*sigmaA(mdm,logsigma,131.0)*integrate.quad(lambda v: fveldist(v)/v,vmin,vmax)[0])
frecoil=interp1d(ereclist,espectlist,bounds_error=False,fill_value=0)
xenoneffx=[x*10**-6 for x in [1.4000000000000000,2.5200000000000000,3.5466666666666700,5.2266666666666700,6.533333333333330,7.466666666666670,8.400000000000000,9.520000000000000,10.453333333333300,11.573333333333300,12.693333333333300,14.466666666666700,17.360000000000000,24.453333333333300,27.06666666666670,29.773333333333300,31.73333333333330,33.506666666666700,34.81333333333330,36.120000000000000,37.42666666666670,38.45333333333330,39.57333333333330,40.97333333333330,42.37333333333330,43.96000000000000,46.38666666666670,49.186666666666700,52.733333333333300,56.37333333333340]]
xenoneffy=[0.0025798775153107500,0.04034155730533700,0.14739737532808400,0.46020787401574800,0.6407461067366580,0.7436052493438320,0.8044696412948380,0.8401315835520560,0.8590012248468940,0.8736657917760280,0.8799314085739290,0.8840776902887140,0.8881903762029750,0.8858778652668420,0.8816000000000000,0.8752195975503060,0.860462642169729,0.835212598425197,0.7973781277340330,0.7553441819772530,0.694412598425197,0.637688888888889,0.5767629046369210,0.48643219597550300,0.4003009623797030,0.2889672790901140,0.17130918635170600,0.07463727034120750,0.0241371828521435,0.0030306211723534200]
xenonefficiency=interp1d(xenoneffx,xenoneffy,bounds_error=False,fill_value=0)
totalevents=integrate.quad(lambda en: frecoil(en)*xenonefficiency(en),4.9*10**-6,40.9*10**-6)
print("total events in XENON1T is "+str(totalevents[0]))

#below is for surface run
cresstdata=[19.7*10**-9]
with open('cresstevents.dat',"r") as dat_file:
    for row in dat_file.readlines():
        cresstdata.append(float(row)*10**-6)
cresstdata.sort()
cresstdata.append(600*10**-9)
#print("cresst data is")
#print(cresstdata)

#below is for cresst underground run
#cresstdata=[30.1*10**-9]
#with open('C3P1_DetA_AR.dat.txt',"r") as dat_file:
#    for row in dat_file.readlines():
#        cresstdata.append(float(row)*10**-6)
#cresstdata.sort()
#cresstdata.append(16000*10**-9)
##print("cresst data is")
##print(cresstdata)

##supercdmsdata=[16.3*10**-9]
##with open('supercdmsdata.dat',"r") as datfile:
##    for row in datfile.readlines():
##        supercdmsdata.append(float(row)*10**-6)
##supercdmsdata.sort()
##supercdmsdata.append(240*10**-9)
##print("supercdmsdata")
##print(len(supercdmsdata))

detector="cresst"

#exposure in gram-days
nA=6.022*10**23
if detector=="cresst":
    exposure=0.046*nA*(60*60*24)#change to this for surface run
    resolution=3.74*10**-9
#    exposure=5689*nA*(60*60*24)#change to this for below ground run
#    resolution=4.6*10**-9
#if detector=="supercdms":
#    exposure=9.9*nA*(60*60*24)
#    resolution=3.86*10**-9
#print(finalvelocitydist[len(fractionlist)-1])

##the below is redundant?
#fveldist=interp1d([10**-6*i*finalvelocitydist[len(fractionlist)-1][i] for i in range(1,3000)],bounds_error=False,fill_value=0)
fveldist=interp1d([10**-6*i for i in range(1,3000)],finalvelocitydist[len(fractionlist)-1],bounds_error=False,fill_value=0)
ereclist=np.linspace(.1*10**-9,1700*10**-9,1000)
espectlistO=[]
espectlistAr=[]
espectlistW=[]
espectlistCa=[]
espectlistWfake=[]
for erec in ereclist:
    vminO=(erec*(mdm+16.*amu)**2/(2.*16.*amu*mdm**2))**(0.5)
    vminAr=(erec*(mdm+27.*amu)**2/(2.*27.*amu*mdm**2))**(0.5)
    vminCa=(erec*(mdm+40.*amu)**2/(2.*40.*amu*mdm**2))**(0.5)
    vminW=(erec*(mdm+184.*amu)**2/(2.*184.*amu*mdm**2))**(0.5)
#    print(vmin)
    vmax=0.003
    if detector=="cresst":#element mass fraction times speed of light times exposure over nuclear mass times dm density times other stuff
#the following two are for surface run
        espectlistO.append(((3.*16.)/(3.*16.+2.*27.))*(3.0*10**10/(1.0))*(exposure/(16*amu))*(0.5*0.3/mdm)*(mdm+16*amu)**2/(2*16*amu*mdm**2)*sigmaA(mdm,logsigma,16.0)*integrate.quad(lambda v: fveldist(v)/v,vminO,vmax)[0])
        espectlistAr.append(((2.*27.)/(3.*16.+2.*27.))*(3.0*10**10/(1.0))*(exposure/(27*amu))*(0.5*0.3/mdm)*(mdm+27*amu)**2/(2*27*amu*mdm**2)*sigmaA(mdm,logsigma,27.0)*integrate.quad(lambda v: fveldist(v)/v,vminAr,vmax)[0])
#the following three are for underground run
#        espectlistO.append(((4.*16.)/(4.*16.+40.+184.))*(3.0*10**10/(1.0))*(exposure/(16*amu))*(0.5*0.3/mdm)*(mdm+16*amu)**2/(2*16*amu*mdm**2)*sigmaA(mdm,logsigma,16.0)*integrate.quad(lambda v: fveldist(v)/v,vminO,vmax)[0])
#        espectlistW.append(((184.)/(4.*16.+40.+184.))*(3.0*10**10/(1.0))*(exposure/(184*amu))*(0.5*0.3/mdm)*(mdm+184*amu)**2/(2*184*amu*mdm**2)*sigmaA(mdm,logsigma,184.0)*integrate.quad(lambda v: fveldist(v)/v,vminW,vmax)[0])
##        espectlistWfake.append(((184.)/(4.*16.+40.+184.))*(3.0*10**10/(1.0))*(exposure/(184*amu))*(0.5*0.3/mdm)*(mdm+16*amu)**2/(2*184*amu*mdm**2)*sigmaA(mdm,logsigma,184.0)*integrate.quad(lambda v: fveldist(v)/v,vminW,vmax)[0])
#        espectlistCa.append(((40.)/(4.*16.+40.+184.))*(3.0*10**10/(1.0))*(exposure/(40*amu))*(0.5*0.3/mdm)*(mdm+40*amu)**2/(2*40*amu*mdm**2)*sigmaA(mdm,logsigma,40.0)*integrate.quad(lambda v: fveldist(v)/v,vminCa,vmax)[0])
    if detector=="supercdms":
        espectlistO.append((3.0*10**10/(1.0))*(exposure/(28*amu))*(0.5*0.3/mdm)*(mdm+28*amu)**2/(2*28*amu*mdm**2)*sigmaA(mdm,logsigma,28.0)*integrate.quad(lambda v: fveldist(v)/v**2,vminO,vmax)[0])
print("espect is")

frecoilO=interp1d(ereclist,espectlistO,bounds_error=False,fill_value=0)
frecoilAr=interp1d(ereclist,espectlistAr,bounds_error=False,fill_value=0)#surface run
#frecoilW=interp1d(ereclist,espectlistW,bounds_error=False,fill_value=0)#underground run
#frecoilWfake=interp1d(ereclist,espectlistWfake,bounds_error=False,fill_value=0)
#frecoilCa=interp1d(ereclist,espectlistCa,bounds_error=False,fill_value=0)#underground run
#result=np.convolve(frecoil,np.exp(-0.5*(x/(3.74*10**-9))**2))
espectlist2=[]
espectlistconv=[]

#below is for surface run
for erec in ereclist:
    if erec > 10**-8:
        espectlist2.append(np.convolve([(frecoilO(en)+frecoilAr(en)) for en in np.linspace(erec-3*resolution,erec+3*resolution,num=1000)],[(6*resolution/(1000))/(resolution*(2.*3.14159)**.5)*2.718**(-0.5*((x)/(resolution))**2.) for x in np.linspace(-3*resolution,3*resolution,num=1000)],"valid")[0])
    else:
        espectlist2.append(frecoilO(erec)+frecoilAr(erec))
frecoil2=interp1d(ereclist,espectlist2,bounds_error=False,fill_value=0)

#below is for underground run
#for erec in ereclist:
#    if erec > 10**-8:
#        espectlist2.append(np.convolve([(frecoilO(en)+frecoilCa(en)+frecoilW(en)) for en in np.linspace(erec-3*resolution,erec+3*resolution,num=1000)],[(6*resolution/(1000))/(resolution*(2.*3.14159)**.5)*2.718**(-0.5*((x)/(resolution))**2.) for x in np.linspace(-3*resolution,3*resolution,num=1000)],"valid")[0])
#    else:
#        espectlist2.append(frecoilO(erec)+frecoilCa(erec)+frecoilW(erec))
#frecoil2=interp1d(ereclist,espectlist2,bounds_error=False,fill_value=0)

cressteffx=[x*10**-9 for x in [1,15.27,17.89,20.43,23.00,25.58,28.20,30.71,33.20,35.06,1000]]#surface
cressteffy=[.001,.12,.32,.57,.80,.94,.985,.995,.995,.996,1]#surface
#cressteffx=[x*10**-9 for x in [.1,22.1,26.7,30,33.8,38.6,45.6,53.3,68,84.7,112,149,211,310,438,639,1108,1920,3461,5953,9317,12960,15899,20000]]#below ground
#cressteffy=[0.001,.018,.107,.282,.395,.509,.535,.553,.57,.578,.588,.593,.598,.608,.626,.636,.642,.645,.647,.643,.624,.634,.654,.654]#below ground
cresstefficiency=interp1d(cressteffx,cressteffy,bounds_error=False,fill_value=0)
if detector=="cresst":
#these integrals don't really work because the energy range is too wide but they aren't used for computing the CL anyway
    totalevents=integrate.quad(lambda en: frecoil2(en)*cresstefficiency(en),19.7*10**-9,600*10**-9,limit=100)#change to this for surface run
#    totalevents=integrate.quad(lambda en: frecoil2(en)*cresstefficiency(en),30.1*10**-9,16000*10**-9,limit=100)#change to this for below ground run
    print("total events in CRESST is "+str(totalevents[0]))
if detector=="supercdms":
    totalevents=integrate.quad(lambda en: .883*frecoil2(en),16.3*10**-9,240*10**-9,limit=100)
#factor of 2 is to account for Emken et al.'s assumption that half of flux is not blocked by the Earth
#factor of .001 is for subdominant DM
sizelist=[]
for i in range(len(cresstdata)-1):
    sizelist.append(2*integrate.quad(lambda en: frecoil2(en)*cresstefficiency(en),cresstdata[i],cresstdata[i+1],limit=100)[0])

#compute CL for CRESST detector using the optimal interval method
mu1=sum(sizelist)
xmax=max(sizelist)
cl=sum([(1+n/(mu1-n*xmax))*math.exp(-n*xmax)*(n*xmax-mu1)**n/(math.factorial(n)) for n in range(int(mu1/xmax)+1)])
print("cl is "+str(cl))

fig = plt.figure(figsize=(5.5,5))

for j in range(len(fractionlist)):
    plt.plot([10**-6*i for i in range(1,3000)],[finalvelocitydist[j][i] for i in range(len(finalvelocitydist[j]))])

plt.xlabel("Velocity [c]", fontsize=16)
plt.ylabel("Attenuated Velocity PDF",fontsize=16)
plt.yscale("log")
plt.xlim([0.000,0.0028])
plt.ylim([10**-20,10**3])

#compute attenuation using the straight line approximation for comparison (only works for underground run, depth 1400 meters)
velArray=[i*10**-5 for i in range(1,300)]
velArray2=[i for i in velArray]
scatterArray=[velocitydist(v) for v in velArray]
scatterArray2=[i for i in scatterArray]

def dedx(en):
    return 0.5*sum([linvlist[j]*en*elossmaxfrac(mdm,nucleuslist[j]) for j in range(len(nucleuslist))])
def integrand(en):
    return -1.0/(dedx(en))

imin=0
jacmin=0
imax=0
jacmax=0
counter=0
for i in range(len(velArray)):
    if counter>1 and scatterArray[i]==0:
        break
    if scatterArray[i]>0:
        counter=counter+1
        vc=velArray[i]
        func = lambda v: quad(integrand,0.5*vc**2*mdm,0.5*v**2*mdm)[0]-140000
        vnewc=np.abs(fsolve(func,0.0001)[0])
        jacobian=vc/vnewc
        if jacmin==0:
            jacmin=jacobian
            imin=i
        jacmax=jacobian
        imax=i
        velArray2[i]=vnewc
for i in range(len(velArray)-1):
    if scatterArray[i]==0:
        if i<imin:
            velArray2[i]=velArray[i]/jacmin
        if i>imax:
            velArray2[i]=velArray[i]/jacmax
for i in range(len(scatterArray)):
    if scatterArray[i]>0 and i<len(scatterArray)-1:
        scatterArray2[i]=scatterArray[i]*(velArray[i+1]-velArray[i-1])/(velArray2[i+1]-velArray2[i-1])
plt.plot(velArray2,scatterArray2,"k--")

plt.tick_params(which='both',direction='in',labelsize=10)

plt.tight_layout()
plt.savefig("velocitydistribution.pdf")
plt.show()
