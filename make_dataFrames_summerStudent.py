import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from root_numpy import root2array
import glob
from numpy.lib.recfunctions import stack_arrays
import h5py
import json

def root2pandas(files_path, tree_name, **kwargs):
    '''
    Args:
    -----
        files_path: a string like './data/*.root', for example
        tree_name: a string like 'Collection_Tree' corresponding to the name of the folder inside the root 
                   file that we want to open
        kwargs: arguments taken by root2array, such as branches to consider, start, stop, step, etc
    Returns:
    --------    
        output_panda: a pandas dataframe like allbkg_df in which all the info from the root file will be stored
    
    Note:
    -----
        if you are working with .root files that contain different branches, you might have to mask your data
        in that case, return pd.DataFrame(ss.data)
    '''
    # -- create list of .root files to process
    files = glob.glob(files_path)
    
    # -- process ntuples into rec arrays
    ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])

    try:
        return pd.DataFrame(ss)
    except Exception:
        return pd.DataFrame(ss.data)
    
    
def flatten(column):
    '''
    Args:
    -----
        column: a column of a pandas df whose entries are lists (or regular entries -- in which case nothing is done)
                e.g.: my_df['some_variable'] 

    Returns:
    --------    
        flattened out version of the column. 

        For example, it will turn:
        [1791, 2719, 1891]
        [1717, 1, 0, 171, 9181, 537, 12]
        [82, 11]
        ...
        into:
        1791, 2719, 1891, 1717, 1, 0, 171, 9181, 537, 12, 82, 11, ...
    '''
    try:
        return np.array([v for e in column for v in e])
    except (TypeError, ValueError):
        return column


BKG = ['DYBJets-Pt100To200',
       'DYBJets-Pt200ToInf',
       'DYJets-BGenFilter-Pt100To200',
       'DYJets-BGenFilter-Pt200ToInf',
       'DYToLL_HT100to200_madgraph',
       'DYToLL_HT1200to2500_madgraph',
       'DYToLL_HT200to400_madgraph',
       'DYToLL_HT2500toInf_madgraph',
       'DYToLL_HT400to600_madgraph',
       'DYToLL_HT600to800_madgraph',
       'DYToLL_HT800to1200_madgraph',
       'DYToLL_M4to50_HT100to200_madgraph',
       'DYToLL_M4to50_HT200to400_madgraph',
       'DYToLL_M4to50_HT400to600_madgraph',
       'DYToLL_M4to50_HT600toInf_madgraph',
       'DYToLL_madgraph',
       'QCD_HT1000to1500',
       'QCD_HT1500to2000',
       'QCD_HT2000toInf',
       'QCD_HT200to300',
       'QCD_HT300to500',
       'QCD_HT500to700',
       'QCD_HT700to1000',
       'ST_s-c_4f_lep_PSw',
       'ST_t-c_antitop_4f_inc',
       'ST_t-c_top_4f_inc',
       'ST_tW_antitop_5f_inc',
       'ST_tW_top_5f_inc_PSw',
       'TT_AllHadronic',
       'TT_DiLep',
       'TT_SingleLep',
       'WBJets-Pt100To200',
       'WBJets-Pt200ToInf',
       'WJets_BGenFilter-Pt100To200',
       'WJets_BGenFilter-Pt200ToInf',
       'WJets-HT100To200',
       'WJets-HT1200To2500',
       'WJets-HT200To400',
       'WJets-HT400To600',
       'WJets-HT600To800',
       'WJets-HT800To1200',
       'WJets_madgraph',
       'WW',
       'WW_1L1Nu2Q',
       'WW_LNu2Q_nlo',
       'WZ',
       'ZBJetsToNuNu_Pt-100to200',
       'ZBJetsToNuNu_Pt-200toInf',
       'ZJetsToNuNu_BGenFilter_Pt-100to200',
       'ZJetsToNuNu_BGenFilter_Pt-200toInf',
       'ZJetsToNuNu_HT100To200',
       'ZJetsToNuNu_HT200To400',
       'ZJetsToNuNu_HT2500ToInf',
       'ZJetsToNuNu_HT400To600',
       'ZJetsToNuNu_HT600To800',
       'ZJetsToNuNu_HT800To1200',
       'ZZ',
       'ZZ_2L2Q_nlo',
       'ZZ_4L',
       'QCDbEnr_HT100to200',] 

SIGNAL = ['WplusH125_powheg',
          'WminusH125_powheg',
          'WminusH_HToBB_WToQQ',]
trainfiles_BKG = []
testfiles_BKG = []
trainfiles_SIG = []
testfiles_SIG = []



branches= ['recon_Wmass', 'recon_Hmass', 'deltaR_reconHW', 'recon_Hpt', 'recon_Wpt', 'deltaR_bb', 'deltaR_W', 'Jet_HT', 'bJet_pt1', 'bJet_pt2', 'WJet_pt1', 'WJet_pt2', 'bJet_btagDeepB1', 'bJet_btagDeepB2', 'WJet_btagDeepB1', 'WJet_btagDeepB2' , 'weight']
#selection = ['isWenu', 'isWmunu', 'controlSample', 'Pass_nominal', 'twoResolvedJets', 'cutFlow', 'sampleIndex', 'usingBEnriched', 'nAddJets302p5_puid', 'event']
selection = ['event']


list_train_df_bkg = []
list_test_df_bkg = []
list_train_df_sig = []
list_test_df_sig = []
input_dir = '/nfs/dust/cms/user/hkaveh/workarea/DNN/prepareData/'
for key in BKG :
    try:
        trainfiles_BKG =glob.glob(input_dir + '/*%s*' % (key))
#        testfiles_BKG = glob.glob('/nfs/dust/cms/user/hkaveh/VHbbAnalysisNtuples/vhbb_2017_me/haddjobs/bdt/*%s*_Odd*' % (key))
        for f in trainfiles_BKG:
            print(f)
            try:
                df = root2pandas(f,'Events', branches=branches+selection)
                #df = df[(df['isWenu'] | df['isWmunu']) & (df['controlSample'] == 0) & (df['V_pt'] >= 150) & (df['V_pt'] < 250) & (df['H_mass'] > 90) & (df['H_mass'] < 150) & (df['hJets_btagged_0']>0.8001) & (df['hJets_btagged_1']>0.1522) & df['Pass_nominal'] & df['twoResolvedJets'] & (df['cutFlow']>=2) & (df['nAddJets302p5_puid']==0) & (df['event'] % 2 == 0)]
                df = df[(df['event'] % 2 == 0)]
                df = df[branches]
                list_train_df_bkg.append(df)
                print(df.head())
            except:
                continue
        for f in trainfiles_BKG:
            print(f)
            try:
                df = root2pandas(f,'Events', branches=branches+selection)
                #df = df[(df['isWenu'] | df['isWmunu']) & (df['controlSample'] == 0) & (df['V_pt'] >= 150) & (df['V_pt'] < 250) & (df['H_mass'] > 90) & (df['H_mass'] < 150) & (df['hJets_btagged_0']>0.8001) & (df['hJets_btagged_1']>0.1522) & df['Pass_nominal'] & df['twoResolvedJets'] & (df['cutFlow']>=2) & (df['nAddJets302p5_puid']==0) & (df['event'] % 2 != 0)]
                df = df[(df['event'] % 2 != 0)]
                df = df[branches]
                list_test_df_bkg.append(df)
                print(df.head())
            except:
                continue

    except:
        continue
    


for key in SIGNAL :
    trainfiles_SIG = glob.glob(input_dir + '/*%s*' % (key))
    for f in trainfiles_SIG:
        print(f)
        df = root2pandas(f,'Events', branches=branches+selection)
        #df = df[(df['isWenu'] | df['isWmunu']) & (df['controlSample'] == 0) & (df['V_pt'] >= 150) & (df['V_pt'] < 250) & (df['H_mass'] > 90) & (df['H_mass'] < 150) & (df['hJets_btagged_0']>0.8001) & (df['hJets_btagged_1']>0.1522) & df['Pass_nominal'] & df['twoResolvedJets'] & (df['cutFlow']>=2) & (df['nAddJets302p5_puid']==0) & (df['event'] % 2 == 0)]
        df = df[(df['event'] % 2 == 0)]
        df = df[branches]
        list_train_df_sig.append(df)
        print(df.head())
    for f in trainfiles_SIG:
        print(f)
        df = root2pandas(f,'Events', branches=branches+selection)
        #df = df[(df['isWenu'] | df['isWmunu']) & (df['controlSample'] == 0) & (df['V_pt'] >= 150) & (df['V_pt'] < 250) & (df['H_mass'] > 90) & (df['H_mass'] < 150) & (df['hJets_btagged_0']>0.8001) & (df['hJets_btagged_1']>0.1522) & df['Pass_nominal'] & df['twoResolvedJets'] & (df['cutFlow']>=2) & (df['nAddJets302p5_puid']==0) & (df['event'] % 2 != 0)]
        df = df[(df['event'] % 2 != 0)]
        df = df[branches]
        list_test_df_sig.append(df)
        print(df.head())



#train_branches= ['H_mass', 'H_pt', 'V_pt', 'SA5', 'V_mass', 'MET_Pt', 'hJets_leadingPt', 'hJets_subleadingPt', 'jjVPtRatio', 'HJ1_HJ2_dEta', 'HVdPhi', 'hJets_btagged_0', 'hJets_btagged_1','Top1_mass_fromLepton_regPT_w4MET']
train_branches= ['recon_Wmass', 'recon_Hmass', 'deltaR_reconHW', 'recon_Hpt', 'recon_Wpt', 'deltaR_bb', 'deltaR_W', 'Jet_HT', 'bJet_pt1', 'bJet_pt2', 'WJet_pt1', 'WJet_pt2', 'bJet_btagDeepB1', 'bJet_btagDeepB2', 'WJet_btagDeepB1', 'WJet_btagDeepB2' ]


featureTrain_df_sig = pd.concat([i for i in list_train_df_sig])
weightTrain_df_sig = featureTrain_df_sig['weight']
featureTrain_df_sig = featureTrain_df_sig[train_branches]
print('concated:', featureTrain_df_sig.head())
targetTrain_df_sig = []
targetTrain_df_sig = np.zeros(featureTrain_df_sig.shape[0])
print(targetTrain_df_sig)

featureTrain_df_bkg = pd.concat([i for i in list_train_df_bkg])
weightTrain_df_bkg = featureTrain_df_bkg['weight']
featureTrain_df_bkg = featureTrain_df_bkg[train_branches]
print('concated:', featureTrain_df_bkg.head())
targetTrain_df_bkg = []
targetTrain_df_bkg = np.ones(featureTrain_df_bkg.shape[0])
print(targetTrain_df_bkg)

featuresTrain = pd.concat((featureTrain_df_sig, featureTrain_df_bkg))
weightsTrain = pd.concat((weightTrain_df_sig, weightTrain_df_bkg))
targetsTrain =  np.concatenate((targetTrain_df_sig, targetTrain_df_bkg), axis=0)
#featuresTrain = featureTrain_df_sig
#targetsTrain = targetTrain_df_sig
#weightsTrain = weightTrain_df_sig



#featuresTrain, featuresTest, targetsTrain, targetsTest, weightsTrain, weightsTest = train_test_split(featuresTrain_.as_matrix(), targetsTrain_, weightsTrain_.as_matrix(), train_size=0.5)


featureTest_df_sig = pd.concat([i for i in list_test_df_sig])
weightTest_df_sig = featureTest_df_sig['weight']
featureTest_df_sig = featureTest_df_sig[train_branches]
print('concated:', featureTest_df_sig.head())
targetTest_df_sig = []
targetTest_df_sig = np.zeros(featureTest_df_sig.shape[0])
print(targetTest_df_sig)

featureTest_df_bkg = pd.concat([i for i in list_test_df_bkg])
weightTest_df_bkg = featureTest_df_bkg['weight']
featureTest_df_bkg = featureTest_df_bkg[train_branches]
print('concated:', featureTest_df_bkg.head())
targetTest_df_bkg = []
targetTest_df_bkg = np.ones(featureTest_df_bkg.shape[0])
print(targetTrain_df_bkg)

featuresTest = pd.concat((featureTest_df_sig, featureTest_df_bkg))
targetsTest =  np.concatenate((targetTest_df_sig, targetTest_df_bkg), axis=0)
weightsTest = pd.concat((weightTest_df_sig, weightTest_df_bkg))
#featuresTest = featureTest_df_sig
#targetsTest = targetTest_df_sig
#weightsTest = weightTest_df_sig

print(featuresTest, targetsTest, weightsTest)


outputFileName = "data_summer.h5"


data = {
        'train': {
            'X': featuresTrain.values,
            'y': targetsTrain,
            'sample_weight': weightsTrain.values,
            },
        'test': {
            'X': featuresTest.values,
            'y': targetsTest,
            'sample_weight': weightsTest.values,
            },
        'category_labels': {0: 'SIG', 1: 'BKG'},
        'meta': {
            'variables': ' '.join(train_branches),
            }
        }
# write output file
f = h5py.File(outputFileName, 'w')
for k in ['meta', 'category_labels']:
    f.attrs[k] = json.dumps(data[k].items())
for k in ['train', 'test']:
    for k2 in data[k].keys():
        f.create_dataset(k + '/' + k2, data=data[k][k2], compression="gzip", compression_opts=9)
f.close()

 
