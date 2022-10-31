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
    vars = ['e','et','eta','phi','deta','dphi']
    d = { key:[] for key in vars }
    d['sampling']=[]
    d['eventNumber']=[]
    if nov < 0 or nov > event.GetEntries():
        nov = event.GetEntries()
    
    for entry in progressbar( range(nov) , event.GetEntries() , prefix='Reading...') : 
        event.GetEntry(entry) 

        cell_cont = event.retrieve("CaloCellContainer_Cells")
        det_cont = event.retrieve("CaloDetDescriptorContainer_Cells")

        for caloCell in cell_cont:
            det_link = getattr(caloCell,"descriptor_link")
            for key in vars:
                d[key].append( getattr(caloCell,key) )
            det = det_cont.at(det_link)
            d['sampling'].append(getattr(det,'sampling'))
            d['eventNumber'].append(entry)


    
    return DataFrame(d) 

def read_events( path , nov=-1):
    
    event = EventStore(path, "physics")
    vars = ['et','eta','phi','reta','rphi','rhad','eratio','weta2','f1','f3']
    d = { key:[] for key in vars }
    d['rings'] = []
    
    if nov < 0 or nov > event.GetEntries():
        nov = event.GetEntries()
    
    for entry in progressbar( range(nov) , event.GetEntries() , prefix='Reading...') : 
        event.GetEntry(entry) 
        
        cluster_cont = event.retrieve("CaloClusterContainer_Clusters")
        
        for caloRings in event.retrieve("CaloRingsContainer_Rings"):
            
            emClus = cluster_cont.at(caloRings.cluster_link)
            for key in vars:
                d[key].append( getattr(emClus,key) ) 

            d['rings'].append(stdvector_to_list(caloRings.rings))
    
    return DataFrame(d)

# path = '/home/juan.marin/lorenzetti/files/Zee_rp/AOD/Zee.AOD.root'
path = '/home/juan.marin/lorenzetti/files/pileup/Zee_mb/AOD/Zee_mb.AOD.root'
zee = read_events(path )
# zee_cells = zee_cells.loc[abs(zee_cells['eta']) <= 2.47]
# path = '/home/juan.marin/lorenzetti/files/JF17/AOD/JF17.AOD.root'
path = '/home/juan.marin/lorenzetti/files/pileup/JF17_mb/AOD/JF17_mb.AOD.root'
jet = read_events(path)
# jet_cells = jet_cells.loc[abs(jet_cells['eta']) <= 2.47]
import numpy as np
features = np.array(['avgmu', 'L2Calo_ring_0', 'L2Calo_ring_1', 'L2Calo_ring_2',
       'L2Calo_ring_3', 'L2Calo_ring_4', 'L2Calo_ring_5', 'L2Calo_ring_6',
       'L2Calo_ring_7', 'L2Calo_ring_8', 'L2Calo_ring_9',
       'L2Calo_ring_10', 'L2Calo_ring_11', 'L2Calo_ring_12',
       'L2Calo_ring_13', 'L2Calo_ring_14', 'L2Calo_ring_15',
       'L2Calo_ring_16', 'L2Calo_ring_17', 'L2Calo_ring_18',
       'L2Calo_ring_19', 'L2Calo_ring_20', 'L2Calo_ring_21',
       'L2Calo_ring_22', 'L2Calo_ring_23', 'L2Calo_ring_24',
       'L2Calo_ring_25', 'L2Calo_ring_26', 'L2Calo_ring_27',
       'L2Calo_ring_28', 'L2Calo_ring_29', 'L2Calo_ring_30',
       'L2Calo_ring_31', 'L2Calo_ring_32', 'L2Calo_ring_33',
       'L2Calo_ring_34', 'L2Calo_ring_35', 'L2Calo_ring_36',
       'L2Calo_ring_37', 'L2Calo_ring_38', 'L2Calo_ring_39',
       'L2Calo_ring_40', 'L2Calo_ring_41', 'L2Calo_ring_42',
       'L2Calo_ring_43', 'L2Calo_ring_44', 'L2Calo_ring_45',
       'L2Calo_ring_46', 'L2Calo_ring_47', 'L2Calo_ring_48',
       'L2Calo_ring_49', 'L2Calo_ring_50', 'L2Calo_ring_51',
       'L2Calo_ring_52', 'L2Calo_ring_53', 'L2Calo_ring_54',
       'L2Calo_ring_55', 'L2Calo_ring_56', 'L2Calo_ring_57',
       'L2Calo_ring_58', 'L2Calo_ring_59', 'L2Calo_ring_60',
       'L2Calo_ring_61', 'L2Calo_ring_62', 'L2Calo_ring_63',
       'L2Calo_ring_64', 'L2Calo_ring_65', 'L2Calo_ring_66',
       'L2Calo_ring_67', 'L2Calo_ring_68', 'L2Calo_ring_69',
       'L2Calo_ring_70', 'L2Calo_ring_71', 'L2Calo_ring_72',
       'L2Calo_ring_73', 'L2Calo_ring_74', 'L2Calo_ring_75',
       'L2Calo_ring_76', 'L2Calo_ring_77', 'L2Calo_ring_78',
       'L2Calo_ring_79', 'L2Calo_ring_80', 'L2Calo_ring_81',
       'L2Calo_ring_82', 'L2Calo_ring_83', 'L2Calo_ring_84',
       'L2Calo_ring_85', 'L2Calo_ring_86', 'L2Calo_ring_87',
       'L2Calo_ring_88', 'L2Calo_ring_89', 'L2Calo_ring_90',
       'L2Calo_ring_91', 'L2Calo_ring_92', 'L2Calo_ring_93',
       'L2Calo_ring_94', 'L2Calo_ring_95', 'L2Calo_ring_96',
       'L2Calo_ring_97', 'L2Calo_ring_98', 'L2Calo_ring_99', 
       'et', 'eta', 'phi', 'eratio', 'reta', 'rphi', 'f1',
       'f3', 'rhad', 'weta2'], dtype='<U31')
etBins = np.array([15,20,30,40,50,1000000])
etaBins = np.array([0,0.8,1.37,1.54,2.37,2.50])


et_bins = [[15*GeV,20*GeV],[20*GeV,30*GeV],[30*GeV,40*GeV],[40*GeV,50*GeV],[50*GeV,1000000*GeV]]
eta_bins = [[0,0.8],[0.8,1.37],[1.37,1.54],[1.54,2.37],[2.37,2.5]]

for etaIdx ,eta  in enumerate(eta_bins):
    for etIdx,et in enumerate(et_bins):
        data =[]
        target=[]
        zee_bin = zee.loc[ (abs(zee['eta']) >= eta[0]) & (abs(zee['eta']) < eta[1]) & (zee['et'] >= et[0]) & (zee['et'] < et[1])]
        for index, electron in zee_bin.iterrows():
            single_zee = [0]
            for ring in electron['rings']: single_zee.append(ring)
            for key in features[101:]: single_zee.append(electron[key])
            data.append(single_zee)
            target.append(1)
        
        jet_bin = jet.loc[ (abs(jet['eta']) >= eta[0]) & (abs(jet['eta']) < eta[1]) & (jet['et'] >= et[0]) & (jet['et'] < et[1])]
        for index,  jets in jet_bin.iterrows():
            single_jet = [0]
            for ring in jets['rings']: single_jet.append(ring)
            for key in features[101:]: single_jet.append(jets[key])
            data.append(single_jet)
            target.append(0)
        
        data = np.array(data)
        target = np.array(target)
        etBinIdx = np.array(etIdx)
        etaBinIdx = np.array(etaIdx)
        filename = 'lorenzetti_Pileup60_electron_jet_et'+str(etIdx)+'_eta'+str(etaIdx)
        print('Saving ' + filename)
        np.savez(filename,features=features, data=data, target=target,etBinIdx=etBinIdx,etaBinIdx=etaBinIdx, etBins=etBins,etaBins=etaBins)

