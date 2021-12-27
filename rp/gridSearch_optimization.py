#!/usr/bin/env python
# coding: utf-8

# In[1]:


from lorenzetti_utils.read_events import *
from Gaugi import load
from Gaugi.monet.utils import getColor,getColors
from Gaugi.monet.PlotFunctions import *
from Gaugi.monet.TAxisFunctions import *
from Gaugi.monet.AtlasStyle import *
from Gaugi import stdvector_to_list, progressbar
from ROOT import TCanvas, TH1F, TH1I, TFile
from ROOT import TLatex, gPad
from ROOT import kRed, kBlue, kBlack,TLine,kBird, kOrange,kGray, kYellow, kViolet, kGreen, kAzure
from pprint import pprint
from pandas import DataFrame


import array
import numpy as np
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
SetAtlasStyle()
GeV=1000.


# In[2]:


def read_cells( path,nov=-1 ):
    
    event = EventStore(path, "physics")
    layer_dict = { 
                    0 : 'PS',
                    1 : 'PS',
                    2 : 'EM1',
                    3 : 'EM2',
                    4 : 'EM3',
                    5 : 'HAD1',
                    6 : 'HAD2',
                    7 : 'HAD3',
                    8 : 'HAD1',
                    9 : 'HAD2',
                    10: 'HAD3',
                    11: 'EM1',
                    12: 'EM2',
                    13: 'EM3',
                    14: 'HAD1',
                    15: 'HAD2',
                    16: 'HAD3',
                }
    vars = ['e','et','eta','phi','deta','dphi']
    d = { key:[] for key in vars }
    d['sampling']=[]
    d['eventNumber']=[]
    d['layer']=[]
    d['roi']=[]
    if nov < 0 or nov > event.GetEntries():
        nov = event.GetEntries()
    
    for entry in progressbar( range(nov) , event.GetEntries() , prefix='Reading...') : 
        event.GetEntry(entry)
        truth_cont = event.retrieve("TruthParticleContainer_Particles")
        print("Number of RoI's: " + str(len(truth_cont)))

        cell_cont = event.retrieve("CaloCellContainer_Cells")
        det_cont = event.retrieve("CaloDetDescriptorContainer_Cells")
        rings_cont = event.retrieve("CaloRingsContainer_Rings")
        cluster_cont = event.retrieve("CaloClusterContainer_Clusters")
        for roi, caloRings in enumerate(event.retrieve("CaloRingsContainer_Rings")):
            emClus = cluster_cont.at(caloRings.cluster_link)
            clus_eta = getattr(emClus,'eta')
            clus_phi = getattr(emClus,'phi')
            for caloCell in cell_cont:
                etaCell = getattr(caloCell,'eta')
                phiCell = getattr(caloCell,'phi')
                if abs(etaCell - clus_eta) > 0.2 or abs(phiCell - clus_phi) > 0.2: continue
                det_link = getattr(caloCell,"descriptor_link")
                for key in vars:
                    d[key].append( getattr(caloCell,key) )
                det = det_cont.at(det_link)
                d['sampling'].append(getattr(det,'sampling'))
                d['eventNumber'].append(entry)
                d['layer'].append(layer_dict[getattr(det,'sampling')])
                d['roi'].append(roi)


    
    return DataFrame(d) 


# In[3]:


def LZTLabel(canvas, x,y,text,color=1):
    l = TLatex()
    l.SetNDC()
    l.SetTextFont(72)
    l.SetTextColor(color)
    delx = 0.17*696*gPad.GetWh()/(472*gPad.GetWw())
    l.DrawLatex(x,y,"Lorenzetti")
    p = TLatex()
    p.SetNDC()
    p.SetTextFont(42)
    p.SetTextColor(color)
    p.DrawLatex(x+delx,y,text)
    canvas.Update()


# In[4]:


def getCellFromSampling(dataframe, layer):
    return dataframe.loc[(dataframe['layer'] == layer)]
    # return dataframe.loc[((dataframe['sampling'] == sampling[0]) | (dataframe['sampling'] == sampling[1]) )]


# In[5]:


def getSamplingsFromLayer(layer):
    layer_dict = {'PS'  : [0,1],
                  'EM1' : [2,11],
                  'EM2' : [3,12],
                  'EM3' : [4,13],
                  'HAD1': [14,5,8],
                  'HAD2': [15,6,9],
                  'HAD3': [16,7,10]
    }
    return layer_dict[layer]


# In[6]:


def calculateCellDistance(eta_hot, phi_hot, eta, phi):
    from math import pi as pi
    from math import sqrt
    dphi = abs(phi - phi_hot)
    if dphi > pi:
        dphi = 2*pi - abs(phi - phi_hot)
    deta = abs(eta - eta_hot)
    return sqrt(dphi*dphi + deta*deta)


# In[7]:


def calculateRp(dataframe, alpha=1, beta=0):
    hotEta = dataframe['eta'][dataframe['et'].idxmax()]
    hotPhi = dataframe['phi'][dataframe['et'].idxmax()]
    event = dataframe['eventNumber'].values[0]
    # if event == 9:
        # print("Iteration: ---- > et: %f" % ()) 
        # print(dataframe)

    num = 0
    den = 0
    for index, cell in dataframe.iterrows():
        if den < 0 or num < 0:
            print("Iteration: ---- > et: %f |ri: %f | num: %f | den: %f" % (cell['et'],ri,num,den)) 
        den += cell['et']**alpha if cell['et'] > 0 else 0
        ri = calculateCellDistance( hotEta, hotPhi, cell['eta'],cell['phi'])
        num += (cell['et']**alpha) * (ri**beta) if cell['et'] > 0 else 0
        event = cell['eventNumber']
        if ri > 2:
            print("Iteration: ---- > et: %f |ri: %f | num: %f | den: %f" % (cell['et'],ri,num,den)) 
    if den == 0:
        return -1
    return num/den


# In[8]:


def calculateSP(signal, background, threshold):
    import numpy as np
    from math import sqrt
    tp = sum(signal <= np.ones(len(signal))*threshold)
    fn = sum(signal > np.ones(len(signal))*threshold)
    tn = sum(background >= np.ones(len(background))*threshold)
    fp = sum(background<np.ones(len(background))*threshold)
    pd = tp / (tp + fn)
    pf = fp / (tn + fp)
    sp = sqrt(sqrt(pd*(1-pf)) * ((pd + (1-pf))/2))
    return sp, pd, pf


# In[9]:


def propagate(zee_cells, jet_cells, events,layer_list, alpha, beta):
    # rp_sgn = {layer:[] for layer in layer_list }
    # rp_bkg = {layer:[] for layer in layer_list }
    rp_sgn = []
    rp_bkg = []
    for event in range(events):
        zee_event = zee_cells.loc[(zee_cells['eventNumber'] == event)]
        jet_event = jet_cells.loc[(jet_cells['eventNumber'] == event)]
        # for layer in layer_list:
        #     samplings = getSamplingsFromLayer(layer)

        #     zee_layered = getCellFromSampling(zee_event, layer)
        #     jet_layered = getCellFromSampling(jet_event, layer)
        # if zee_layered.empty or jet_layered.empty: continue
        # if calculateRp(zee_layered, alpha, beta) < 0 or calculateRp(jet_layered, alpha, beta) < 0: continue
        # rp_sgn[layer].append(calculateRp(zee_layered, alpha, beta))
        # rp_bkg[layer].append(calculateRp(jet_layered, alpha, beta))
        if zee_event.empty or jet_event.empty: continue
        print(zee_event['roi'].max())
        for roi in range(zee_event['roi'].max()+1):
            zee_roi = zee_event.loc[(zee_event['roi'] == roi)]
            print(zee_roi)
            sgn = calculateRp(zee_roi, alpha, beta)
        # jet_roi = jet_event.loc[(jet_event['roi'] == roi)]
        bkg = calculateRp(jet_event, alpha, beta)
        if sgn < 0 or bkg < 0: continue
        rp_sgn.append(sgn)
        rp_bkg.append(bkg)
        
    return rp_sgn, rp_bkg


# In[10]:


def getMaxSP(rp_sgn, rp_bkg):
#     layer_list = list(rp_sgn.keys())
    import numpy as np
    npoints = 1000

    sp = []
    pd = []
    pf = []
    # max_rp_sgn = np.max(rp_sgn[layer])
    # max_rp_bkg = np.max(rp_bkg[layer])
    max_rp_sgn = np.max(rp_sgn)
    max_rp_bkg = np.max(rp_bkg)
    m_max = np.max([max_rp_sgn,max_rp_bkg])
    thresholds = np.arange(0,m_max, m_max/npoints)
    print("Iteration: ---- > max_sgn: %f |max_bkg: %f | m_max: %ff" % (max_rp_sgn,max_rp_bkg,m_max))
    for thr in thresholds:
        # sp.append(calculateSP(rp_sgn[layer],rp_bkg[layer], thr))
        # sp.append(calculateSP(rp_sgn,rp_bkg, thr))
        temp_sp, temp_pd, temp_pf = calculateSP(rp_sgn,rp_bkg, thr)
        sp.append(temp_sp)
        pd.append(temp_pd)
        pf.append(temp_pf)
    max_sp = np.max(sp)
    threshold = thresholds[np.argmax(sp)]
    max_pd = pd[np.argmax(sp)]
    max_pf = pf[np.argmax(sp)]

    return max_sp, threshold, max_pd, max_pf


# In[11]:


path = '/home/juan.marin/Zee.AOD.root'
# path = '/home/edmar.egidio/LZT/Optimization/Zee.AOD.root'
zee_cells = read_cells(path, nov=10 )
zee_cells = zee_cells.loc[abs(zee_cells['eta']) <= 2.47]
path = '/home/juan.marin/JF17.AOD.root'
# path ='/home/edmar.egidio/LZT/Optimization/JF17.AOD.root'
jet_cells = read_cells(path, nov=10)
jet_cells = jet_cells.loc[abs(jet_cells['eta']) <= 2.47]


# In[15]:


layer_list = ['PS','EM1','EM2','EM3','HAD1','HAD2','HAD3']

# result_dict = {layer : {'alpha': [], 'beta': [], 'max_sp':[], 'thr': []} for layer in layer_list}
result_dict = {'alpha': [], 'beta': [], 'max_sp':[], 'thr': [], 'max_pd': [], 'max_pf': []}

import numpy as np
# from tempfile import TemporaryFile
events = 10
alpha = np.arange(0,0.5,0.1)
beta  = np.arange(0,0.5,0.1) 
alp = 0
rp_sgn = []
rp_bkg = []
for i in alpha:
    alp = alp + 1
    bet = 0
    for k in beta:
        bet = bet + 1        
        # for layer in layer_list:
        rp_sgn, rp_bkg = propagate(zee_cells, jet_cells, events,layer_list, alp, bet)
        # rp_sgn = rp_sgn + sgn
        # rp_bkg = rp_bkg + bkg

        max_sp, thr, max_pd, max_pf = getMaxSP(rp_sgn, rp_bkg)
        result_dict['max_sp'].append(max_sp)
        result_dict['thr'].append(thr)
        result_dict['alpha'].append(i)
        result_dict['beta'].append(k)
        result_dict['max_pd'].append(max_pd)
        result_dict['max_pf'].append(max_pf)

        # result_dict[layer]['max_sp'].append(max_sp)
        # result_dict[layer]['thr'].append(thr)
        # result_dict[layer]['alpha'].append(i)
        # result_dict[layer]['beta'].append(k)
        
        print("Iteration: ---- > alpha: %f | beta: %f | SP: %f | PD: %f | PF: %f" % (i, k,max_sp, max_pd, max_pf)) 

np.savez('rpOptimization',result_dict=result_dict)
# In[16]:


# print('Alpha: %f' %( result_dict['EM2']['alpha'][np.argmax(result_dict['EM2']['max_sp'])]))
# print('Beta: %f'  %( result_dict['EM2']['beta'][np.argmax(result_dict['EM2']['max_sp'])]))

